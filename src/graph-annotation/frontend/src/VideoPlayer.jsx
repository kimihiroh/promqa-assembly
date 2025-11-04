interface VideoPlayerProps {
    videoSrc: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoSrc }) => {
    return (
        <video controls style={{ width: "100%", height: "40vh" }}>
            <source src={videoSrc} type="video/mp4" />
        </video>
    );
};

export default VideoPlayer;
