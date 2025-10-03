import os
import aiohttp
import aiofiles
import mimetypes
from typing import Optional
from urllib.parse import urlparse
from config.setting import env

TEMP_DIR = "temp"

class UrlUploader:

    @staticmethod
    async def download_media(url: str) -> Optional[str]:
        """Download media from BSP URL to temporary file"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Get file extension from content-type or URL
                        content_type = response.headers.get('content-type', '')
                        url_path = urlparse(url).path
                        
                        # Try to get extension from content-type first
                        extension = mimetypes.guess_extension(content_type) or ''
                        
                        # If no extension from content-type, try to get from URL
                        if not extension and '.' in url_path:
                            extension = os.path.splitext(url_path)[1]
                        
                        # Generate temp filename with extension
                        random_id = os.urandom(8).hex()
                        filename = os.path.join(TEMP_DIR, f"temp_{random_id}{extension}")
                        
                        # Save file
                        async with aiofiles.open(filename, 'wb') as f:
                            await f.write(await response.read())
                        
                        return filename
                    else:
                        return None
        except Exception as e:
            return None

    @staticmethod
    async def upload_file(filepath: str) -> Optional[str]:
        """Upload file to Siloam server"""
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            
            # Determine content type based on file extension
            filename = os.path.basename(filepath)
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            
            # Use aiofiles for async file operations
            async with aiofiles.open(filepath, 'rb') as f:
                file_data = await f.read()
                data.add_field('files[]', 
                            file_data,
                            filename=filename,
                            content_type=content_type)  # Add content_type here
                
            data.add_field('uploader', 'assets')

            async with aiohttp.ClientSession() as session:
                async with session.post(env.base_url_uploader, data=data) as response:
                    print(f"response: {await response.text()}")
                    if response.status == 200:
                        result = await response.json()
                        # Get first file URL from response
                        if result.get('data') and len(result['data']) > 0:
                            return result['data'][0]['uri']
                    
                    return None
                    
        except Exception as e:
            return None
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    @staticmethod
    async def process_media(url: str) -> Optional[str]:
        """Process media from BSP URL to Siloam URL"""
        try:
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)
                
            # Download media
            temp_file = await UrlUploader.download_media(url)
            if not temp_file:
                return None

            # Upload to Siloam
            return await UrlUploader.upload_file(temp_file)

        except Exception as e:
            return None
