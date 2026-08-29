### allowed types ###

ALLOWED_TYPES = {
    'image': {
        'resource_type': 'image',
        'extensions': {'jpg', 'jpeg', 'png', 'webp'},
        'mime_types': {'image/jpeg', 'image/png', 'image/webp'},
        'max_size': 5 * 1024 * 1024,
    },
    'audio': {
        'resource_type': 'video',
        'extensions': {'mp3', 'wav', 'ogg', 'flac'},
        'mime_types': {'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac'},
        'max_size': 50 * 1024 * 1024,
    },
    'video': {
        'resource_type': 'video',
        'extensions': {'mp4', 'mov', 'webm', 'mkv'},
        'mime_types': {'video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska'},
        'max_size': 200 * 1024 * 1024,
    },
    'document': {
        'resource_type': 'raw',
        'extensions': {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt'},
        'mime_types': {
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',        # .xlsx
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'text/plain',
        },
        'max_size': 20 * 1024 * 1024,
    },
}


### get_file_extension ###

import os

def get_file_extension(filename):
    """
    Gives the file extension.

    Args:
        filename (str): File name

    Returns:
        str: File extension
    """
    
    if not '.' in filename:
        return None
    
    _, ext = os.path.splitext(filename)

    if not ext:
        return None
    
    ext = ext[1:].lower()

    return ext