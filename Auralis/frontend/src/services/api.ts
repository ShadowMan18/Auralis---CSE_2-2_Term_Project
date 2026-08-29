import api from './ApiService.ts';

interface ApiResponse {
    message: string;
}

//=== add apis here ===//
export async function demoApiCall(param: FormData): Promise<ApiResponse> {
    return api.request<ApiResponse>('/api/demo', {
        method: 'POST',
        body: param
    });
}