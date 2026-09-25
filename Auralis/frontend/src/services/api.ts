import api from './ApiService.ts';

export interface PathPrediction {
    species: string;
    confidence: number;
}

export interface UploadSampleResponse {
    threshold: number;
    paths: {
        dsp: PathPrediction[];
        cnn: PathPrediction[];
        pann: PathPrediction[];
    };
    path_errors?: Partial<Record<'dsp' | 'cnn' | 'pann', string>>;
}

export async function uploadSample(uploadFormData: FormData): Promise<UploadSampleResponse> {
    return await api.request('/api/upload-sample', {
        method: 'POST',
        body: uploadFormData
    });
}
