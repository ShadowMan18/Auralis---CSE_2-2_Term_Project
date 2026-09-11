import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';

export function Home() {
    const navigate = useNavigate();

    return (
        <div id='home_page'>
            <center>
                <h1>Welcome to Auralis!</h1>
                <Button onClick={() => navigate('/auth/signup')}>Sign up</Button>
                <Button onClick={() => navigate('/auth/login')}>Login</Button>
            </center>
        </div>
    )
}