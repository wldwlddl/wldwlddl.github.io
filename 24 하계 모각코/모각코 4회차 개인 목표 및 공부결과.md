#### 목표:
- 파이토치 학습
	- dataset&dataloader
	- transform
	

#### 결과
###### dataset&dataloader
- 데이터셋 코드와 학습 코드는 분리하는 게 좋다(유지보수가 어려울 수 있기 때문)
- Pytorch의 FashionMNIST와 같은 다양한 데이터셋을 제공한다. 
```
from torch.utils.data import Dataset  
from torchvision.transforms import ToTensor  
from torchvision import datasets  
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(  
    root="data",  
    train=True,  
    download=True,  
    transform=ToTensor()  
)

test_data = datasets.FashionMNIST(  
    root="data",  
    train=False,  
    download=True,  
    transform=ToTensor()  
)
```
- 첫 번째 단락의 코드는 필요한 모듈을 임포트 하는 과정이고 두번째 단락은 FashionMNIST 데이터셋을 로딩하고 pytorch에서 사용할 수 있도록 텐서 형태로 변환하여 training_data변수에 저장하는 역할이다. 마지막 단락은 FashionMNIST데이터셋의 테스트 데이터를 로딩하고 이를 test_data변수에 저장하는 역할을 한다. 

_개인적으로 실습을 진행하다 세 번째 단락의 train =True라고 적는 실수를 저질렀는데, False라고 적어야하는 이유는 Ture라 적으면 학습 데이터가 들어오게 되어 과적합이 일어날 수 있기 때문이라고 한다. 모델이 학습한 데이터를 학습한 결과를 평가할 때 사용하는 데이터인 테스트 데이터로 사용하게 되면 객관적인 평가를 할 수 없다는 문제점도 가지고 있다._

- 아래의 코드를 통해 데이터를 시각화할 수 있다. 이 코드를 실행하면 T-shirt, Trouser 등의 10개의 패션 아이템 중 랜덤으로 9개의 패션 아이템을 뽑는 코드이다.
```
labels_map = {  
    0: "T-shirt",  
    1: "Trouser",  
    2: "pullover",  
    3: "Dress",  
    4: "Coat",  
    5: "Sandal",  
    6: "shirt",  
    7: "Sneaker",  
    8: "Bag",  
    9: "Ankle Boot",  
}  
figure = plt.figure(figsize=(8, 8))  
cols, rows = 3, 3  
for i in range(1, cols*rows + 1):  
    sample_idx = torch.randint(len(training_data),size=(1,)).item()  
    img, label = training_data[sample_idx]  
    figure.add_subplot(rows, cols, i)  
    plt.title(labels_map[label])  
    plt.axis("off")  
    plt.imshow(img.squeeze(), cmap="gray")  
    plt.show()
```

- dataloader는 샘플들을 미니배치(전체 데이터를 N등분하여 각각의 학습 데이터를 배치 방식((한번에 큰 묶음으로 데이터를 입력받는것))으로 학습) 하고 매 에폭(인공신경망에서 전체 데이터셋에 대한 한번 학습을 완료한 상태)마다 데이터를 섞어 과적합을 막고 멀티프로세싱(다수의 프로세서가 다수의 작업을 함께 처리하는 것)을 사용해 검색속도를 높이려고 하는 모든 과정을 간단한 API(소프트웨어 애플리케이션이 서로 통신하여 데이터, 특징 및 기능을 교환할 수 있도록 한 일련의 규칙)로 이러한 복잡한 과정들을 추상화한 객체이다. (iterable하다)
```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

###### 변형(transform)
- 이 작업은 데이터를 조작하고 학습에 적합하게 만든다.  
- TorchVision 데이터셋들은 데이터를 변형할 수 있도록 두 개의 주요 매개변수인 transform과  target_transform을 제공한다. transform은 feature(특징- 예를 들어 이미지 데이터라면 크기 조정, 정규화, 텐서 등이 있음)을 변경하기 위한 매개변수로 입력 데이터를 학습에 적합한 형태로 변형한다. target_transform은 레이블(정답)을 변경하기 위한 매개변수이다. 레이블을 원-핫 인코딩(원-핫 인코딩: 범주형 데이터를 이진 벡터로 변환하는 방법 예를 들어 고양이와 개가 있다고 하면 고양이는 (1, 0), 개는 (0, 1)으로 둘 수 있다.)이나 라벨 스무딩( 라벨 스무딩: 원-핫 인코딩과 비슷한 방법이나 좀 더 자세하다. 예를 들어 원-핫 인코딩에서 고양이는 (1, 0), 개는 (0, 1)이였다면 라벨 스무딩을 이용하면 고양이는 (0.9, 0.1), 개는(0.1, 0.9)로 표현할 수 있다. ) 등의 변형을 지원한다.




#### 후기 
실습에서 오류가 난 줄 모르고 그대로 진행했다가 보고 따라한 코드와 달라 비교해봤을 때 틀렸다는 걸 인지하자 궁금증이 생겼다. 왜 오류가 나지 않았지? 검색해보고 내가 이 코드를 잘못 이해하고 있었음과 train을 어떨 때 True를 써야 하는지 알게 되었다. 이 경험을 통해 여태까지 내가 실습할 때 코드를 잘 이해하지 않고 그냥 따라만 쓰지는 않았나 성찰하는 시간을 가졌다.