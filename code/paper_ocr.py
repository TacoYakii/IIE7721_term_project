import pdf2image
from glob import glob 
import os 
import easyocr
from tqdm import tqdm 


def convert_to_image(pdf_path:str, sv_root:str) -> None: 
    """
    Convert every pdf files to image files from given pdf_path

    Args:
        pdf_path (str): pdf files root path 
        sv_path (str): Save root path 

    """
    pdf_files = list(glob(f"{pdf_path}/*.pdf"))
    for pdf in tqdm(pdf_files, desc="Converting pdf to image"): 
        with open(pdf, "rb") as f: 
            img = pdf2image.convert_from_bytes(f.read())
        img_nm = os.path.basename(pdf).split(".")[0] 
        sv_path = os.path.join(sv_root, f"{img_nm}.jpg") 
        img[0].save(sv_path, "JPEG")


def image_to_txt(img_path:str, sv_root:str) -> None: 
    """
    Convert every image files to text files from given img_path

    Args:
        img_path (str): image root path 
        sv_path (str): save root path 
    """
    reader = easyocr.Reader(
        ["en", "ko"], 
        gpu=True 
    )
    img_dir_list = list(glob(f"{img_path}/*.jpg"))
    for img_dir in tqdm(img_dir_list, desc="converting image to txt"): 
        with open(img_dir, "rb") as f: 
            img = f.read()
        result = reader.readtext(
            img, 
            detail=0, 
            paragraph=True
            )
        img_nm = os.path.basename(img_dir).split(".")[0] 
        sv_path = os.path.join(sv_root, f"{img_nm}.txt") 
        with open(sv_path, "w", encoding="utf-8") as f: 
            for line in result: 
                words = line.split() 
                f.write("\n".join(words)+"\n") 
                

original_pdf_path = "/home/taco/Documents/projects/paper_analysis/data/original" 
image_sv_root = "/home/taco/Documents/projects/paper_analysis/data/image"
txt_sv_root = "/home/taco/Documents/projects/paper_analysis/data/txt"
convert_to_image(original_pdf_path, image_sv_root)
image_to_txt(image_sv_root, txt_sv_root)