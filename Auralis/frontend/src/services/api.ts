import api from './ApiService.ts';

interface UploadSampleResponse {
    species: string[];
    confidence: number[];
    decision?: string | null;
    geo_prior_applied?: boolean;
}

export async function uploadSample(uploadFormData: FormData): Promise<UploadSampleResponse> {
    return await api.request('/api/upload-sample', {
        method: 'POST',
        body: uploadFormData
    });
}
