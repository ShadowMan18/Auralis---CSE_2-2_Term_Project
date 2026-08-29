const API_BASE_URL = 'http://localhost:8000';

class ApiService {

    async request<T = any>(endpoint: string, options: RequestInit = {}, isRetry = false): Promise<T> {
        try {
            const token = localStorage.getItem('token');

            const headers: Record<string, string> = { ...options.headers as Record<string, string> };

            if (!(options.body instanceof FormData)) {
                headers['Content-Type'] = 'application/json';
            }

            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                ...options,
                headers,
                credentials: 'include',
            });

            if (response.status === 401 && !isRetry && endpoint !== '/api/auth/refresh') {
                const refreshed = await this.refreshToken();

                if (refreshed) {
                    return this.request<T>(endpoint, options, true);
                }

                localStorage.removeItem('token');
                window.location.href = '/auth/login';
                throw new Error('Session expired. Please log in again.');
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || errorData.message || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        }
        catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    private async refreshToken(): Promise<boolean> {
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
                method: 'POST',
                credentials: 'include',
            });

            if (!response.ok) return false;

            const data = await response.json();
            localStorage.setItem('token', data.token);
            return true;
        }
        catch (error) {
            console.error('Token refresh failed:', error);
            return false;
        }
    }

}

export default new ApiService();