import api from './ApiService.ts';

interface UploadSampleResponse {
    species: string[],
    confidence: number[]
}

export async function uploadSample(uploadFormData: FormData): Promise<UploadSampleResponse> {
    return await api.request('/api/upload-sample', {
        method: 'POST',
        body: uploadFormData
    });
}