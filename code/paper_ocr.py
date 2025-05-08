import pdf2image
from glob import glob 
import os 
import easyocr
from tqdm import tqdm 

def create_dir(path): 
    if not os.path.exists(path): 
        os.makedirs(path) 
    
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
        sv_path = os.path.join(sv_root, f"{img_nm}") 
        create_dir(sv_path) 
        for i in range(len(img)): 
            img[i].save(os.path.join(sv_path, f"{i}.jpg"), "JPEG")


def image_to_txt(img_path:str, sv_root:str) -> None: 
    """
    Convert every image files to text files from given img_path

    Args:
        img_path (str): image root path 
        sv_path (str): save root path 
    """
    create_dir(sv_root)
    reader = easyocr.Reader(
        ["en", "ko"], 
        gpu=True 
    )
    img_dir_list = list(glob(f"{img_path}/*"))
    for img_dir in tqdm(img_dir_list, desc="converting image to txt"): 
        file_nm = os.path.basename(img_dir).split(".")[0] 
        img_list = list(glob(f"{img_dir}/*.jpg")) 
        create_dir(os.path.join(sv_root, file_nm))
        for page in img_list: 
            page_no = os.path.basename(page).split(".")[0] 
            with open(page, "rb") as f: 
                page_img = f.read() 
            page_str = reader.readtext(page_img, detail=0, paragraph=True) 
            with open(os.path.join(sv_root, file_nm, f"{page_no}.txt"), "w", encoding="utf-8") as txt_file: 
                for line in page_str: 
                    txt_file.write(line)


original_pdf_path = "/home/taco/Documents/projects/paper_analysis/data/original" 
image_sv_root = "/home/taco/Documents/projects/paper_analysis/data/image"
txt_sv_root = "/home/taco/Documents/projects/paper_analysis/data/txt"
convert_to_image(original_pdf_path, image_sv_root)
image_to_txt(image_sv_root, txt_sv_root)