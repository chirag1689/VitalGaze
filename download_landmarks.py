import requests
import bz2
import os

def download_landmark_detector():
    print("Downloading facial landmark predictor...")
    
    # URL for the file
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        # Download the compressed file
        response = requests.get(url)
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as f:
            f.write(response.content)
        
        print("Decompressing file...")
        # Decompress the file
        with open("shape_predictor_68_face_landmarks.dat", "wb") as new_file:
            with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
        
        # Remove the compressed file
        os.remove("shape_predictor_68_face_landmarks.dat.bz2")
        print("Facial landmark predictor ready!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    download_landmark_detector()