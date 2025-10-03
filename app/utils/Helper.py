import asyncio
import json
from typing import Any, Dict, List, Tuple, Union
import uuid
import requests
from PIL import Image
from io import BytesIO
import base64
from google.cloud import storage
from config.setting import env
from config.credentials import google_credential

credentials = google_credential()
gcs_client = storage.Client(project=env.google_project_name, credentials=credentials)
gcs_bucket = gcs_client.get_bucket(env.bucket_name)

async def prepare_images_for_llm(
        images_or_urls: List[Union[str, bytes]],
        text_content: str = "",
        use_compression: bool = False,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Upload images if needed and return (content_parts, url_list).
        - content_parts → ready to send to LLM
        - url_list → the uploaded or provided URLs
        """
        # print(f"{text_content}")
        # If bytes → upload to GCS
        content_parts: List[Dict] = []
        
        if images_or_urls and isinstance(images_or_urls[0], (bytes, bytearray)):
            content_parts, url_list = await upload_images_to_gcs(images_or_urls)
            images_or_urls = url_list
            
        if use_compression:
            content_parts: List[Dict] = []
            url_list = images_or_urls
            for url in url_list:
                compression_tasks = [asyncio.create_task(_compress_image_async(url))]
                base64_results = await asyncio.gather(*compression_tasks) if compression_tasks else []
                for base64_img in base64_results:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"} 
                    })
        else:
            url_list = images_or_urls
            content_parts = create_image_content_from_urls(url_list)
            if text_content:
                content_parts.append({"type": "text", "text": text_content})
        
        return content_parts, url_list

async def _compress_image_async(url: str) -> str:
    # print("_compress_image_async")
    # print(url)
    """Compress image asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compress_image_sync, url)
    except Exception as e:
        print(f'Error in _compress_image_async: {e}')
        raise e

def _compress_image_sync(image_url: str) -> str:
    # print("_compress_image_sync")
    # print(image_url)
    """Synchronous image compression."""
     
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image: {response.status_code}")

    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")

    compressed_io = BytesIO()
    image.save(compressed_io, format="JPEG", optimize=True, quality=86)
    compressed_io.seek(0)
        
    return base64.b64encode(compressed_io.read()).decode("utf-8")

def create_image_content_from_urls(urls: List[str], add_text_placeholder: bool = True) -> List[Dict]:
    # print("create_image_content_from_urls")
    """Create image content parts from URLs."""
    content_parts = []
    for url in urls:
        content_parts.append({"type": "image_url", "image_url": {"url": url}})
        
    if add_text_placeholder:
        content_parts.append({"type": "text", "text": ""})
            
    return content_parts

async def _upload_single_image(blob, file_bytes: bytes, content_type: str):
    # print("_upload_single_image")
    """Upload a single image asynchronously."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: blob.upload_from_string(file_bytes, content_type=content_type)
    )
    await loop.run_in_executor(None, blob.make_public)
    
async def upload_images_to_gcs(files: List[bytes], content_type: str = "image/png/jpg") -> Tuple[List[Dict], List[str]]:
    # print("upload_images_to_gcs")
    """Upload images to GCS and return content parts and URLs."""
    if not gcs_bucket:
        raise ValueError("GCS bucket must be configured to use upload_images_to_gcs")
        
    content_parts = []
    url_list = []
        
    # Create upload tasks
    upload_tasks = []
    for file_bytes in files:
        unique_id = uuid.uuid4().hex
        file_name = f"image_{unique_id}.{content_type.split('/')[-1]}"
        blob = gcs_bucket.blob(file_name)
        task = asyncio.create_task(_upload_single_image(blob, file_bytes, content_type))
        upload_tasks.append((task, blob))
        
    # Execute uploads
    for task, blob in upload_tasks:
        await task
        url = blob.public_url
        content_parts.append({"type": "image_url", "image_url": {"url": url}})
        url_list.append(url)
    content_parts.append({"type": "text", "text": " "})
    return content_parts, url_list

def _postprocess_response(ai_msg: Any, state: Dict[str, Any]) -> Tuple[Dict, Any]:
    """
    Extract JSON from the LLM response, update with computed fields,
    and return (parsed_json, updated_ai_msg).
    """
    # content_str = getattr(ai_msg, "content", str(ai_msg))
    try:
        # if isinstance(ai_msg, str):
        #     parsed_json = json.loads(ai_msg)
        # elif isinstance(ai_msg, dict):
        #     parsed_json = ai_msg
        # else:
        #     raise TypeError(f"Unsupported type for ai_msg: {type(ai_msg)}")
        parsed_json = ai_msg

        # Default safe values
        total_item = parsed_json.get("total_item", 1) or 1
        primary = state.get("primary")
        first = primary.get("first_nested_unit", 1) or 1
        second = primary.get("second_nested_unit", 1) or 1
        third = primary.get("third_nested_unit", 1) or 1

        total_item_quantity = total_item * first * second * third
        parsed_json["total_item_quantity"] = total_item_quantity
        parsed_json["total_item"] = total_item

        # print("parsed json: ", parsed_json)
        # print("ai msg before: ", ai_msg)
        # parsed_msg = json.dumps(parsed_json)
        # print("Ai msg after: ", ai_msg)
        # print(ai_msg)
        return parsed_json

    except Exception as e:
        print("JSON postprocess failed:", e)
        return ai_msg

PRICING = {
    "gemini-2.0-flash": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gemini-2.5-flash": {
        "input": 0.30 / 1_000_000,
        "output": 2.50 / 1_000_000,
    },
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 3.00 / 1_000_000,   # $0.003 / 1k
        "output": 15.00 / 1_000_000, # $0.015 / 1k
    },
}

def compute_usage(response: dict, model: str) -> dict:
    """Extract tokens and compute cost based on model pricing."""
    # Try Gemini-style first
    usage = response.get("response_metadata", {}).get("usage_metadata", {})
    
    prompt_tokens = usage.get("prompt_token_count")
    completion_tokens = usage.get("candidates_token_count")
    total_tokens = usage.get("total_token_count")

    # Fallback: top-level usage_metadata
    if prompt_tokens is None or completion_tokens is None:
        usage_top = response.get("usage_metadata", {})
        prompt_tokens = prompt_tokens or usage_top.get("input_tokens")
        completion_tokens = completion_tokens or usage_top.get("output_tokens")
        total_tokens = total_tokens or usage_top.get("total_tokens")

    # 🔹 Extra Fallback for Claude schema
    if prompt_tokens is None or completion_tokens is None:
        usage_claude = response.get("response_metadata", {}).get("usage", {})
        prompt_tokens = prompt_tokens or usage_claude.get("prompt_tokens")
        completion_tokens = completion_tokens or usage_claude.get("completion_tokens")
        total_tokens = total_tokens or usage_claude.get("total_tokens")

    # Defaults
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total_tokens = total_tokens or (prompt_tokens + completion_tokens)

    rates = PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        prompt_tokens * rates["input"] +
        completion_tokens * rates["output"]
    )

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": round(cost, 6),
    }




