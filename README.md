# mlops_landmark
## Описание задачи

### Формальная задача:
Хотим по фото понимать, соотносится ли оно с каким-то фото из нашего сета документов. Если да, то с каким

### Домен фотографий: 
Архитектурное/интерьерное окружение (фото зданий)

### Применение: 
Социальная сеть. Автоматическое проставление геотегов фотографий на основании их содержания, основываясь на списке интересующих мест. На основании этой инфо - продажа таргетной рекламы.

## Данные
Данные для бейзлайна предоставлены датасетом от google  Google Landmarks Dataset v2 (GLDv2). Очищенная версия для оценки подходов предоставленна на kaggle: [link](https://www.kaggle.com/c/landmark-retrieval-2021/data) 

\+ Данный датасет обладает прекрасной сбалансированностью по земному шару, предоставляя различные архитектурные стили, фото с различных ракурсов и при разном освещении, архитектура и интерьер. В целом датасет сформирован опытными ds для соревнования описанной выше задаче.(это честно я так пишу, llm-free text) 

\- Россия почти не представлена в датасете, в рамках локального продукта придется расширить обучающую выборку

## Решение
В качестве базового решения пока предлагается рассмотреть голову над эмбеддингами, сгенерированными обученной на классификацию моделью
* head: 1 layer fully connected
* backbone: [mobile net](https://huggingface.co/timm/tf_mobilenetv3_small_100.in1k)

библиотеки, за незнанием продвинутого инструментария: pytorch, sklearn, albumentations (аугментация изображений)
## Архитектура применения в продакшене:

![image](https://github.com/GalaxyDragon/mlops_landmark/assets/22980159/9e607da4-d194-46f2-897d-0443396fc6a3)
### Текстовое краткое описание:
параллельно происходит:
* инференс на проде, развешивание лейблы заведений к фотографиям
* переобучение, чтобы добавить новые заведения в список лейблов
* после достижения преемлемых метрик модель на проде заменяется свежеобученной
В рамках курса нас интересует реализация 2 пункта, если правильно понимаю

задача вдохновлена: [arxiv dataset paper](https://arxiv.org/pdf/2004.01804.pdf)
решение: [kaggle](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-glret-2021)
