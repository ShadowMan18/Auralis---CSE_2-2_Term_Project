import api from '../ApiService.ts';

type Category = 'image' | 'video' | 'audio' | 'document';

// upload file 

interface PresignedUploadResponse {
    signature: string;
    timestamp: number;
    api_key: string;
    cloud_name: string;
    resource_type: string;
    public_id: string;
}

export async function uploadFile(file: File, category: Category): Promise<string> {
    const presigned = await api.request<PresignedUploadResponse>('/api/cloud/presigned-upload', {
        method: 'POST',
        body: JSON.stringify({
            filename: file.name,
            content_type: file.type,
            category
        })
    });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('api_key', presigned.api_key);
    formData.append('timestamp', String(presigned.timestamp));
    formData.append('signature', presigned.signature);
    formData.append('public_id', presigned.public_id);

    const uploadUrl = `https://api.cloudinary.com/v1_1/${presigned.cloud_name}/${presigned.resource_type}/upload`;

    const uploadResult = await fetch(uploadUrl, {
        method: 'POST',
        body: formData
    });

    if (!uploadResult.ok) {
        throw new Error('Upload to cloud failed');
    }

    try {
        await api.request('/api/cloud/confirm-upload', {
            method: 'POST',
            body: JSON.stringify({ key: presigned.public_id, category })
        });
    }
    catch {
        throw new Error('Upload to cloud failed');
    }

    return presigned.public_id;
}