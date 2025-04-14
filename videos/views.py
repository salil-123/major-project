import time
import os
import re
import cv2
import dlib
import torch
import torch.nn as nn
import torchvision
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView
from django.http import JsonResponse
from PIL import Image as pImage
from tqdm import tqdm
from .forms import UploadForm
from .models import Videos_Post
from videos.models_deep import model_selection
from videos.transform import xception_default_data_transforms, resnet18_default_data_transforms
from videos.transform_meso import mesonet_data_transforms
from videos.classifier import Meso4

num_progress = 0
frame_progress = 0
face_progress = 0
DetectImg = []
DetectPrediction = []
dictionaryProgress = {}
dictionaryProgress1 = {}
dictionaryProgress2 = {}

def index(request):
    return render(request, "index.html")

class UserVideosView(LoginRequiredMixin, ListView):
    model = Videos_Post
    paginate_by = 6
    template_name = "process_videos.html"

    def get_queryset(self):
        return Videos_Post.objects.all().order_by('?')

class UploadVideosView(LoginRequiredMixin, CreateView):
    model = Videos_Post
    form_class = UploadForm
    template_name = "upload.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("user_videoshow", kwargs={"username": self.request.user.username})

def VideosInformationView(request, pk):
    models = Videos_Post.objects.get(pk=pk)
    videos = "/media/" + str(models.videos)
    dic = {
        "title": models.title,
        "videos": videos,
    }
    return render(request, "process_detail.html", dic)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        size_bb = max(minsize, size_bb)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, modelname):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if modelname == 'XceptionNet':
        preprocess = xception_default_data_transforms['test']
    elif modelname == 'MesoInceptionNet':
        preprocess = mesonet_data_transforms['test']
    elif modelname == 'ResNet18':
        preprocess = resnet18_default_data_transforms['test']
    preprocessed_image = preprocess(pImage.fromarray(image))
    return preprocessed_image.unsqueeze(0)

def predict_with_model(modelname, image, model, post_function=nn.Softmax(dim=1)):
    preprocessed_image = preprocess_image(image, modelname)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)
    prediction = int(prediction.cpu().numpy())
    if modelname == "ResNet18":
        prediction = 1 - prediction
    return prediction, output

def funs(request, pk, m):
    time_start = time.time()
    global num_progress, frame_progress, face_progress, DetectImg, DetectPrediction
    modelname, pk = m, pk

    obj = Videos_Post.objects.get(title=pk)
    videos = "/media/" + str(obj.videos)
    forging_method = obj.forging_method
    compressed_format = obj.compressed_format
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), videos[1:])
    video_path = re.sub(r'\\', r'/', video_path)
    video_fn = os.path.splitext(os.path.basename(video_path))[0] + '.mp4'
    output_path = "A:/major project/FaceForensics-Detection_Website-master/media/in_out_videos/result"
    detect_path = "/media/in_out_videos/result/" + video_fn
    frame_extract = []
    face_frame = []

    model_paths = {
        'XceptionNet': "A:/major project/FaceForensics-Detection_Website-master/faceforensics++_models/xception/xce.pth",
        'MesoInceptionNet': "A:/major project/FaceForensics-Detection_Website-master/faceforensics++_models/Mesonet/mesoinception.pth",
        'ResNet18': "A:/major project/FaceForensics-Detection_Website-master/faceforensics++_models/resnet18/resnet.pth"
    }

    if modelname == 'XceptionNet':
        model, *_ = model_selection(modelname, num_out_classes=2)
        model.load_state_dict(torch.load(model_paths[modelname], map_location=torch.device("cpu")))
    elif modelname == "MesoInceptionNet":
        model = nn.DataParallel(Meso4())
        model.load_state_dict(torch.load(model_paths[modelname], map_location=torch.device('cpu')), strict=False)
    elif modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = torch.load(model_paths[modelname], map_location=torch.device('cpu'))

    reader = cv2.VideoCapture(video_path)
    fourcc = int(cv2.VideoWriter_fourcc(*'H264'))
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    face_detector = dlib.get_frontal_face_detector()
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    print("<=== | Started Videos Splitting | ===>")
    frames = []
    for _ in tqdm(range(num_frames)):
        ret, image = reader.read()
        if not ret:
            break
        frames.append(image)

    sequence_length = 60
    for i in range(1, sequence_length + 1):
        frame = frames[i]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pImage.fromarray(image, 'RGB')
        image_name = f"{video_fn[:-4]}_preprocessed_{i}.png"
        image_path = f"A:/major project/FaceForensics-Detection_Website-master/FF_Detection/preprocess_images/{image_name}"
        img.save(image_path)
        frame_extract.append(image_name)
    print("<=== | Videos Splitting Done | ===>")

    frame_progress = 1
    face_progress = 1
    print("<=== | Started Face Cropping and Predicting Each Frame | ===>")

    for i, image in enumerate(tqdm(frames)):
        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(output_path, video_fn), fourcc, fps, (width, height))

        num_progress = i / num_frames
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)

        if faces:
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            if i < 60:
                image1 = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                img = pImage.fromarray(image1, 'RGB')
                image_name = f"{video_fn[:-4]}_cropped_faces_{i}.png"
                image_path = f"A:/major project/FaceForensics-Detection_Website-master/FF_Detection/preprocess_images/{image_name}"
                img.save(image_path)
                face_frame.append(image_name)

            prediction, output = predict_with_model(modelname, cropped_face, model)

            x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
            label = 'fake' if prediction == 0 else 'real'
            color = (0, 0, 255) if prediction == 0 else (0, 255, 0)

            output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            outs = output.detach().cpu().numpy()[0]
            cv2.putText(image, f"{output_list} => {label}", (x, y + h + 30), font_face, font_scale, color, thickness, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        DetectPrediction.append(outs)
        image_name = f"{video_fn[:-4]}_detect_faces_{i}.png"
        image_path = f"A:/major project/FaceForensics-Detection_Website-master/FF_Detection/preprocess_images/{image_name}"
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        DetectImg.append(image_name)

        writer.write(image)

    writer.release()
    print("<=== | Face Cropping Each Frame Done | ===>")
    print(f'Finished! Output saved under {output_path}')

    time_end = time.time()
    print('Total time:', time_end - time_start)
    print(frame_extract)
    print(face_frame)
    return render(request, "process_result.html", {
        "preprocessed_images": frame_extract,
        "faces_cropped_images": face_frame,
        "resluts": label,
        "detect_path": detect_path,
        "modelname": modelname,
        "detect_videos": f"/media/in_out_videos/result/{video_fn}",
        "compressed_format": compressed_format,
        "forging_method": forging_method,
        "title": pk
    })

def text(request, pk):
    global num_progress, frame_progress, face_progress
    num_progress = frame_progress = face_progress = 0
    obj = Videos_Post.objects.get(title=pk)
    videos = "/media/" + str(obj.videos)
    dic = {
        "title": obj.title,
        "videos": videos,
        "modles": {"XceptionNet", "MesoInceptionNet", "ResNet18"}
    }
    return render(request, "process_detect.html", dic)

def reminder2(request, num):
    t = round(num_progress * 100, 2)
    dictionaryProgress[num] = t
    data_dict2 = {"num_progress": dictionaryProgress}
    return JsonResponse(data_dict2, safe=False)

def reminder1(request, num):
    dictionaryProgress1[num] = face_progress
    dictionaryProgress2[num] = frame_progress
    data_dict1 = {
        "face_progress": dictionaryProgress1,
        "frame_progress": dictionaryProgress2,
    }
    return JsonResponse(data_dict1, safe=False)

def threshold(request):
    yuzhi = float(request.POST.get('gs', 0)) / 100.0
    res_pre = [i for i, pred in enumerate(DetectPrediction) if yuzhi < float(pred[0]) or yuzhi < float(pred[1])]
    res_img = [DetectImg[i] for i in res_pre]
    return render(request, "models_details.html", {
        "totallen": len(DetectImg),
        "threshold": res_img,
        "p": "frames search for you",
        "len": len(res_pre),
        "yuzhi": yuzhi,
        "detectImg": DetectImg
    })

def ModelsDetailView(request):
    return render(request, "models_details.html", {"detectImg": DetectImg, "totallen": len(DetectImg)})