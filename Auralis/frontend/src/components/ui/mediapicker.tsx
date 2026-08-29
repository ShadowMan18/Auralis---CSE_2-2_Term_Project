import { useRef, useState } from "react"
import { Upload, X, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface MediaPickerProps {
    value?: File | string | null
    onChange?: (file: File | null) => void
    accept?: string
    maxSizeMB?: number
    variant?: "avatar" | "default"
    className?: string
}

export function MediaPicker({
    value,
    onChange,
    accept = "image/*",
    maxSizeMB = 5,
    variant = "default",
    className,
}: MediaPickerProps) {
    const inputRef = useRef<HTMLInputElement>(null)
    const [error, setError] = useState<string | null>(null)

    const previewUrl =
        value instanceof File ? URL.createObjectURL(value) : value || null

    const isImage =
        value instanceof File
            ? value.type.startsWith("image/")
            : typeof value === "string"
                ? /\.(jpe?g|png|gif|webp|svg)$/i.test(value)
                : false

    function handleFile(file: File | undefined) {
        setError(null)
        if (!file) return

        if (maxSizeMB && file.size > maxSizeMB * 1024 * 1024) {
            setError(`File must be under ${maxSizeMB}MB`)
            return
        }

        onChange?.(file)
    }

    function handleRemove(e: React.MouseEvent) {
        e.stopPropagation()
        onChange?.(null)
        if (inputRef.current) inputRef.current.value = ""
    }

    return (
        <div className={cn("flex flex-col gap-2", className)}>
            <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                    e.preventDefault()
                    handleFile(e.dataTransfer.files?.[0])
                }}
                className={cn(
                    "relative flex cursor-pointer items-center justify-center border-2 border-dashed border-muted-foreground/25 bg-muted/30 hover:bg-muted/50 transition-colors overflow-hidden",
                    variant === "avatar"
                        ? "h-24 w-24 rounded-full"
                        : "h-40 w-full rounded-md"
                )}
            >
                {previewUrl ? (
                    <>
                        {isImage ? (
                            <img
                                src={previewUrl}
                                alt="Preview"
                                className="h-full w-full object-cover"
                            />
                        ) : (
                            <div className="flex flex-col items-center gap-1 text-muted-foreground">
                                <FileText className="h-8 w-8" />
                                <span className="text-xs">
                                    {value instanceof File ? value.name : "File selected"}
                                </span>
                            </div>
                        )}
                        <Button
                            type="button"
                            size="icon"
                            variant="destructive"
                            className="absolute right-1 top-1 h-6 w-6"
                            onClick={handleRemove}
                        >
                            <X className="h-3 w-3" />
                        </Button>
                    </>
                ) : (
                    <div className="flex flex-col items-center gap-1 text-muted-foreground">
                        <Upload className="h-6 w-6" />
                        <span className="text-xs">Click or drop file</span>
                    </div>
                )}
            </div>

            <input
                ref={inputRef}
                type="file"
                accept={accept}
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
            />

            {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
    )
}