# app.py
import os
import logging
logging.getLogger("streamlit.web.bootstrap").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
import streamlit as st
import shutil
from PIL import Image, ImageDraw
import ffmpeg
import torch
from segment_anything import sam_model_registry, SamPredictor

@st.cache_data
def load_sam(checkpoint="/app/checkpoints/sam_vit_h_4b8939.pth"):
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    return sam.to("cuda" if torch.cuda.is_available() else "cpu")

def extract_n_frames_ffmpeg(video_path: str, n_frames: int, out_dir: str):
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    # 1) make sure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # 2) probe the video duration (in seconds)
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])

    # 3) for each of the N slots, seek + grab one frame
    for i in range(n_frames):
        # timestamp in seconds for this frame
        t = (i * duration) / n_frames
        (
            ffmpeg
            .input(video_path, ss=t)             # seek to t
            .output(os.path.join(out_dir, f'frame_{i+1:03d}.png'),
                    vframes=1,                    # only one frame
                    format='image2')              # force image sequence
            .overwrite_output()
            .run(quiet=True)
        )

def overlay_mask(frame, mask, color=(255,0,0,100)):
    pil = Image.fromarray(frame)
    overlay = Image.new("RGBA", pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = mask.nonzero()
    for y,x in zip(ys,xs):
        draw.point((x,y), fill=color)
    return Image.alpha_composite(pil.convert("RGBA"), overlay)

def main():    
    st.set_page_config(
    page_title="KineSeg",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed"
)
    
    
    st.sidebar.markdown('''
    Made by :green[Anhurion] using :blue[Segment Anything] and 
    :red[Streamlit]. :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:
    ''')
    
    # add_selectbox = st.sidebar.selectbox(
    # "How would you like to be contacted?",
    # ("Email", "Home phone", "Mobile phone")
    
    
    st.title(":orange[KineSeg]: Segment the :blue[freak] out of Robot Motion")
    # st.divider()
    with st.container(height=None, border=True, key=None):
        col_dev, col_sam = st.columns([0.1, 0.9], gap= "small", vertical_alignment= "center", border= False)
    
        # Determine and display device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cuda":
            col_dev.badge("Running on: CUDA", icon=":material/check:", color="green")
        elif str(device) == "cpu":
            col_dev.badge("Running on: CPU", icon="⚠️", color="orange")
        else:
            col_dev.badge("Torch did not find Cuda or CPU device", icon="❌", color="red")
            
            
        try:
                with st.spinner("Loading Segment Anything Model…", show_time=True):
                    if "predictor" not in st.session_state:
                        sam_device = load_sam()  # your existing @st.cache_data loader
                        st.session_state.predictor = SamPredictor(sam_device)
                col_sam.badge("SAM loaded into memory", icon=":material/check:", color="blue")
        except Exception as e:
                col_sam.badge(f"❌ Failed to load SAM model: {e}", icon=":material/check:", color="red")
    
    
    
    
    st.header("Video Import", divider=True)
    with st.container(height=None, border=True, key=None):
        left, right = st.columns(2, gap= "small", vertical_alignment= "top", border= True)
        # interval = st.number_input("Frame interval (sec)", 0.1, 10.0, 1.0)
        
        
        
        
        uploaded = left.file_uploader("Upload Simulation Video", type=["mp4","mov","avi"])
        
        n = right.number_input("How many frames to extract?", min_value=1, max_value=25, value=1)
        
    
        if uploaded:
            # write uploader to a temp file
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            
            out_dir = f"./frames_{os.path.splitext(uploaded.name)[0]}"
            extract_n_frames_ffmpeg(tmp_path, n, out_dir)
            st.success(f"Saved {n} frames to `{out_dir}/`")

            # (now you can iterate through out_dir/*.png for the rest of your pipeline)
            with st.container(height=None, border=True, key=None):
                st.write("Preview:")
                cols = st.columns(min(n, 5), border= False)
                for idx, fname in enumerate(sorted(os.listdir(out_dir))):
                    img = os.path.join(out_dir, fname)
                    cols[idx % len(cols)].image(img, caption=fname, use_container_width=True)
                    
            st.session_state.out_dir = out_dir
            
    st.header("Video Import", divider=True)
    with st.container(height=None, border=True, key=None):
        left, right = st.columns(2, gap= "small", vertical_alignment= "top", border= True)
        saved = []
        
        predictor = st.session_state.predictor
        out_dir = st.session_state.out_dir
        
        for idx, fname in enumerate(sorted(os.listdir(out_dir))):
            if st.button(f"Segment Frame {idx+1}"):
                    predictor.set_image(fname)
                    masks, *_ = predictor.predict(
                        point_coords=None, box=None, multimask_output=False
                    )
                    ovl = overlay_mask(fname, masks[0])
                    st.image(ovl, caption="Overlay", use_column_width=True)
                    if st.button(f"Accept Frame {idx+1}"):
                        saved.append(ovl.convert("RGB"))



    # if video:
    #     frames = extract_frames(video, interval)
    #     sam_device = load_sam()
    #     predictor = SamPredictor(sam_device)
    #     saved = []
    #     for i, frame in enumerate(frames):
    #         st.header(f"Frame {i+1}/{len(frames)}")
    #         st.image(frame, use_column_width=True)
    #         if st.button(f"Segment Frame {i+1}"):
    #             predictor.set_image(frame)
    #             masks, *_ = predictor.predict(
    #                 point_coords=None, box=None, multimask_output=False
    #             )
    #             ovl = overlay_mask(frame, masks[0])
    #             st.image(ovl, caption="Overlay", use_column_width=True)
    #             if st.button(f"Accept Frame {i+1}"):
    #                 saved.append(ovl.convert("RGB"))
    #     if saved:
    #         cols = st.number_input("Columns in montage", 1, 10, 5)
    #         rows = math.ceil(len(saved)/cols)
    #         w,h = saved[0].size
    #         montage = Image.new("RGB", (cols*w, rows*h))
    #         for idx, img in enumerate(saved):
    #             montage.paste(img, ((idx%cols)*w, (idx//cols)*h))
    #         st.image(montage, caption="Final Montage", use_column_width=True)
    #         buf = montage.tobytes()
    #         st.download_button("Download Montage", montage.tobytes(), "montage.png", "image/png")

if __name__=="__main__":
    main()