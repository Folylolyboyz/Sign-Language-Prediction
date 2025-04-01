import { useState, useCallback, FC, JSX, useEffect, useRef } from "react";
import Webcam from "react-webcam";

interface WebcamFeed {
    showVideo: boolean;
    pauseDetected: boolean;
    updateVideo: () => void;
    handDetection: (data: boolean) => void;
    updatePrediction: (data: string) => void;
    socketUrl: string;
}

const WebcamFeed: FC<WebcamFeed> = ({
    showVideo,
    pauseDetected,
    updateVideo,
    handDetection,
    updatePrediction,
    socketUrl,
}): JSX.Element => {
    const [isWebcamAvailable, setIsWebcamAvailable] = useState(true);

    const [showSocketStatus, setSocketStatus] = useState<string>(
        "Press start video to begin"
    );
    const webcamRef = useRef<Webcam>(null);
    const socketRef = useRef<WebSocket | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(
        document.createElement("canvas")
    );
    const handleUserMediaError = useCallback(() => {
        setIsWebcamAvailable(false);
        updateVideo();
    }, []);

    const captureFrame = useCallback(() => {
        const video = webcamRef.current?.video;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (video && ctx) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(
                (blob) => {
                    if (
                        blob &&
                        socketRef.current?.readyState === WebSocket.OPEN
                    ) {
                        blob.arrayBuffer().then((buffer) => {
                            socketRef.current?.send(buffer);
                        });
                    }
                },
                "image/webp",
                0.8
            );
        }
    }, []);

    useEffect(() => {
        if (!showVideo || pauseDetected) return;

        socketRef.current = new WebSocket(socketUrl);
        socketRef.current.onopen = () => {
            setSocketStatus("Hand detection started 👏");
        };

        // get predicted data from server
        socketRef.current.onmessage = (event) => {
            console.log("Received text message:", event.data);
            updatePrediction(event.data);
        };

        socketRef.current.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
        socketRef.current.onclose = () => {
            setSocketStatus("Hand detection stopped 👎");
        };

        const intervalId = setInterval(captureFrame, 100);
        WebSocket.OPEN ? handDetection(false) : handDetection(true);

        return () => {
            clearInterval(intervalId);
            socketRef.current?.close();
        };
    }, [socketUrl, captureFrame, showVideo, handDetection]);

    return (
        <>
            {isWebcamAvailable && showVideo ? (
                <>
                    <Webcam
                        audio={false}
                        height={720}
                        width={1280}
                        mirrored={true}
                        onError={handleUserMediaError}
                        ref={webcamRef}
                        disablePictureInPicture={true}
                        screenshotFormat="image/jpeg"
                        className="w-full aspect-video bg-background/40 border-1 rounded-2xl"
                    />
                </>
            ) : (
                <div className="w-full bg-background/40 border-1 aspect-video flex justify-center items-center rounded-2xl">
                    <p className="capitalize text-md sm:text-2xl tracking-wide">
                        no video feed available
                    </p>
                </div>
            )}
            <p className="text-lg sm:text-xl tracking-wide mt-4 w-full rounded-2xl py-2 px-4">
                <span className="capitalize font-semibold">status: </span>{" "}
                {showSocketStatus}
            </p>
        </>
    );
};

export default WebcamFeed;
