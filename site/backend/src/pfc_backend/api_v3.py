# -*- coding: utf-8 -*-

# from openai import OpenAI
# import whisper
import pandas as pd
import os
# from perplexity import Perplexity
import ast
import boto3
import requests
import json
import subprocess

def process_file(podoption):
  '''
  Prepares an audio file for fact checking and calls the factcheck function with it.
  Args: podoption - an audio file
  Returns: the results of fact checking the podoption file
  '''

  model = whisper.load_model("small.en")
  transcript = model.transcribe(podoption)["text"]
  return transcript

def probe_audio_format(url: str) -> str:
  """
  Checks that the file at url is an audio file, reading only its headers (no full download).

  Args:
      url (str): URL of the target file.
  Returns:
      str: The container format name reported by ffprobe (e.g. "mp3", "mov").
  Raises:
      ValueError: If the file cannot be probed or has no audio stream.
  """
  result = subprocess.run(
      ["ffprobe", "-v", "error", "-show_entries", "format=format_name:stream=codec_type",
       "-of", "json", url],
      capture_output=True, text=True, timeout=60,
  )
  if result.returncode != 0:
      raise ValueError(f"Could not probe {url}: {result.stderr.strip()}")

  info = json.loads(result.stdout)
  if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
      raise ValueError(f"File at {url} has no audio stream")
  return info["format"]["format_name"].split(",")[0]

def url_to_s3(url: str, bucket: str, key: str, max_seconds: int = 30 * 60) -> None:
  """
  Streams the first max_seconds (default 30 minutes) of an audio file into an s3 bucket.
  ffmpeg reads the URL directly and stops once it has enough audio, so the full file is never downloaded.

  Args:
      url (str): URL of the target file.
      bucket (str): AWS S3 bucket name (use get_whisper_bucket)
      key (str): File name to write in the bucket. Can include child paths inside the parent bucket.
      max_seconds (int): Length to clip the audio to, in seconds.
  Raises:
      ValueError: If the file is not an audio file.
      RuntimeError: If ffmpeg fails while clipping.
  """
  fmt = probe_audio_format(url)

  # Output goes to a non-seekable pipe, so MP4/M4A needs a fragmented container.
  out_args = ["-f", fmt]
  content_type = f"audio/{fmt}"
  if fmt == "mov":
      out_args = ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov"]
      content_type = "audio/mp4"
  elif fmt == "mp3":
      content_type = "audio/mpeg"

  proc = subprocess.Popen(
      ["ffmpeg", "-v", "error", "-i", url, "-t", str(max_seconds),
       "-map", "0:a", "-c", "copy", *out_args, "pipe:1"],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
  )

  s3 = boto3.client("s3")
  try:
      s3.upload_fileobj(proc.stdout, bucket, key, ExtraArgs={"ContentType": content_type})
  finally:
      proc.stdout.close()
      stderr = proc.stderr.read().decode()
      proc.wait()

  if proc.returncode != 0:
      s3.delete_object(Bucket=bucket, Key=key)  # don't leave a truncated/corrupt clip behind
      raise RuntimeError(f"ffmpeg failed clipping {url}: {stderr.strip()}")

def get_whisper_bucket(stack_name: str, output_key: str) -> str:
    """
    Gets the S3 bucket name where the whisper model expects audio files to land.
    """
    cfn = boto3.client("cloudformation")
    stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    
    for output in stack.get("Outputs", []):
        if output["OutputKey"] == output_key:
            return output["OutputValue"]
          
    raise KeyError(f"Output '{output_key}' not found in stack '{stack_name}'")

def process_rss(rss_url: str, max_seconds: int = 30 * 60) -> str:
  '''
  Retrieves the first episode from an RSS link, uploads its audio to S3 for whisper, and returns the resulting transcript.
  Args: rss_url: a link to a RSS stream for a podcast
        max_seconds (int): Length to clip the audio to, in seconds.
  Returns: the transcript text of the first episode from the rss_url
  '''

  import feedparser

  feed = feedparser.parse(rss_url)

  if feed.bozo:
    raise RuntimeError(feed.bozo_exception)

  if not feed.entries:
      raise ValueError("RSS parsed but contains no entries")

  # grab the latest episode only
  episode = feed.entries[0]
  title = episode['title']

  if "enclosures" in episode and episode.enclosures:
      audio_url = episode.enclosures[0].href

      # dump file into AWS BUCKET 
      bucket = get_whisper_bucket("Whisper", "BucketName")
      _ = url_to_s3(url = audio_url, bucket = bucket, key = f"uploads/{title}", max_seconds = max_seconds)

      # whisper writes the transcript asynchronously, so wait for it to appear
      s3 = boto3.client("s3")
      transcript_key = f"transcripts/{title}.txt"
      s3.get_waiter("object_exists").wait(
          Bucket = bucket, Key = transcript_key,
          WaiterConfig = {"Delay": 10, "MaxAttempts": 180},
      )
      return s3.get_object(Bucket = bucket, Key = transcript_key)["Body"].read().decode("utf-8")

  else:
      print("This RSS does not contain an audio enclosure.")
      return


# domains_df = pd.read_csv("filtered_attrs.csv")

# trusted_df = domains_df[domains_df["label"] >= 5] #find domains with a reliability ranking of 5 or 6 on a scale of 1-6

def factcheck(transcript, openai_api_key, perplexity_api_key):

  #claim extraction

  system_prompt_extraction = "Please breakdown the following transcript into independent atomic facts. Ignore basic information about the podcast, " \
  "such as the name and host. Ignore promotional/ad content. Ignore personal anecdotes. Here is one example:" \
  "Input: \"She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.\" " \
  "Output: [ {1: \"She currently stars in Love and Destiny.\"}, {2: \"Love and and Destiny is a romantic comedy series.\"}, {3 : \"Love and Destiny premiered in 2019.\"} ]"

  #initialize client with explicit OpenAI API key
  os.environ['OPENAI_API_KEY'] = openai_api_key
  client = OpenAI(
    organization='org-OJ8ihAvp3ut0vdG4Vnxzk7oR',
    project='proj_VcduimxcQxgSopyyOglx06vQ',
  )

  response = client.chat.completions.create(
    model="gpt-5-mini",
    seed=42,
    messages=[
        {"role": "system", "content": system_prompt_extraction},
        {"role": "user", "content": transcript}
    ]
  )
  generated_text = response.choices[0].message.content.strip() #our list of extracted claims
  extracted_claims = ast.literal_eval(generated_text)

  #more preprocessing

  #make container for each claim, its associated label, any supporting sources

  facts = {"num": [], "extracted_claim": [], "label": [], "sources": []}

  facts["num"] = list(range(1, len(extracted_claims) + 1))

  facts["extracted_claim"] = [list(entry.values())[0].strip() for entry in extracted_claims]

  #fact checking

  #initialize client with explicit Perplexity API key
  client = Perplexity(api_key = perplexity_api_key)

  system_prompt_fact_checking = """
  You are a fact-checker. Please check the accuracy of the provided claim, taking into account any previous claims provided as context. Return one of these ratings:
  true, false, misleading/partially true, unverifiable.
  Do not add any explanations or extraneous formatting.
  Here are 4 examples:

  Input: "Antonio Guterres is the UN Secretary General"
  Output: "true"

  Input: "Many of the Maldivians inducted into the Islamic State have come back to the Maldives"
  Output: "false"

  Input: "The Pakistani state supports or turns a blind eye to acts of terrorism"
  Output: "misleading/partially true"

  Input: "Governments can kill bitcoin by making the economic incentive to use it irrelevant"
  Output: "unverifiable"

  """

  previous_claims_context = [] # initialize an empty list for context

  #for each extracted fact:
  for claim in facts["extracted_claim"]:

    user_prompt = f"Fact to check: {claim}"
    if previous_claims_context:
      # use a sliding window of the last 5 claims to prevent context pollution from unrelated stories
      relevant_context = previous_claims_context[-5:]
      user_prompt = f"Previous claims: {relevant_context}\n\n{user_prompt}"

    while True:
      try:
        completion = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": system_prompt_fact_checking},
            {"role": "user", "content": user_prompt}
        ],
        web_search_options={
          "search_context_size": "medium"
        }
        )

        generated_text = completion.choices[0].message.content

        break

      except (SyntaxError, ValueError) as e: #if llm fails to format properly, call again with same fact
        print(f"Error parsing LLM output for claim '{claim}': {e}. Retrying...")
        continue

    facts["label"].append(generated_text)

    # access the URLs from the search and save as sources
    urls = [result.url for result in completion.search_results]
    annot_urls = []

    # "star" any sources that come from trusted domains
    for url in urls:
      annot_url = url
      for index, row in trusted_df.iterrows():
        if row["url"] in url:
          annot_url = "* " + annot_url
          break
      annot_urls.append(annot_url)

    facts["sources"].append(annot_urls)

    previous_claims_context.append(claim)

  df = pd.DataFrame(facts)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)

  return df