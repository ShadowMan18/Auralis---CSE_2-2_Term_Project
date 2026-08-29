import api from '../ApiService.ts'

type Category = 'image' | 'video' | 'audio' | 'document';

// upload file 

export async function uploadFile(file: File, category: Category): Promise<string> {
    const { url, key } = await api.request<{ url: string; key: string }>('/api/cloud/presigned-upload', {
        method: 'POST',
        body: JSON.stringify({ 
            filename: file.name, 
            content_type: file.type,
            category
        })
    });

    const uploadResult = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': file.type },
        body: file,
    });

    if (!uploadResult.ok) {
        throw new Error('Upload to cloud failed');
    }

    try {
        await api.request('/api/cloud/confirm-upload', {
            method: 'POST',
            body: JSON.stringify({ key, category })
        });
    }
    catch {
        throw new Error('Upload to cloud failed');
    }

    return key;
}