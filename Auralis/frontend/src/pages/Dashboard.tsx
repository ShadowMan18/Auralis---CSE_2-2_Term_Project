import { useState } from 'react';
import api from '@/services/ApiService.ts';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { MediaPicker } from '@/components/ui/mediapicker';
import { uploadSample } from '@/services/api';

const uploadFormSchema = z.object({
    sample: z.instanceof(File)
});

type uploadFormValues = z.infer<typeof uploadFormSchema>;

export function Dashboard() {
    const [serverError, setServerError] = useState<string | null>(null);
    const [species, setSpecies] = useState<string[]>([]);
    const [confidence, setConfidence] = useState<number[]>([]);
    
    const uploadForm = useForm<uploadFormValues>({
        resolver: zodResolver(uploadFormSchema),
        // defaultValues: { name: '' },        // define the defualt values
    })

    async function onSubmit(values: uploadFormValues) {
        setServerError(null);

        // add items to uploadFormData
        const uploadFormData = new FormData();
        if (values.sample) uploadFormData.append('sample', values.sample);

        try {
            const result = await uploadSample(uploadFormData);
            setSpecies(result.species);
            setConfidence(result.confidence);
            uploadForm.reset();
        }
        catch {
            setServerError('Submission failed. Please try again.');
        }
    }

    return (
        <div id='dashboard'>
            <h1><b>Dashboard</b></h1>

            <Form {...uploadForm}>
                <form onSubmit={uploadForm.handleSubmit(onSubmit)} className='space-y-6 max-w-md'>
                    <FormField
                        control={uploadForm.control}
                        name='sample'  
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Sample</FormLabel>  
                                <FormControl>
                                    <MediaPicker
                                        value={field.value}
                                        onChange={field.onChange}
                                        accept='audio/*'    
                                    />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {serverError && (
                        <p className='text-sm text-destructive'>{serverError}</p>
                    )}

                    <Button type='submit' disabled={uploadForm.formState.isSubmitting}>
                        {uploadForm.formState.isSubmitting ? 'Uploading...' : 'Upload'}  
                    </Button>
                </form>
            </Form>

            <ul>
                {
                    species.map((s, i) => (
                        <li key={i}>
                            {s}: {confidence[i]}
                        </li>
                    ))
                }
            </ul>
        </div>
    )
}