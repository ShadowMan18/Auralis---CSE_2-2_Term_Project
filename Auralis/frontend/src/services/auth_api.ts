import api from './ApiService.ts'

// signup

interface SignUpResponse {
    user_id: number;
    token: string;
}

export async function signup(formData: FormData): Promise<SignUpResponse> {
    return api.request<SignUpResponse>('/api/auth/signup', {
        method: 'POST',
        body: formData
    });
}

// onboarding

export async function onboarding(formData: FormData) {
    return api.request('/api/auth/onboarding', {
        method: 'POST',
        body: formData
    });
}

// login

interface LoginResponse {
    user_id: number;
    name: string;
    profile_picture: string;
    token: string;
}

export async function login(formData: FormData): Promise<LoginResponse> {
    return api.request<LoginResponse>('/api/auth/login', {
        method: 'POST',
        body: formData
    });
}

export async function logout() {
    return api.request('/api/auth/logout', { method: 'POST' });
}

