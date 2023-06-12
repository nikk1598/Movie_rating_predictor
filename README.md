# Movie_rating_predictor

Docker-image можно скачать через командную строку командой "docker pull niks1598/mrp". Скорее всего, нужно будет в настройках выделить доп. ресурсы для контейнера (т.к. скрипты требуют слишком много памяти).

Или самостоятельно скачать папку с проектом,  затем установить библиотеки, указанные в requirements.txt и запустить из папки frontend через терминал командой "streamlit run main.py --server.maxUploadSize=1028

В github представлен "чистый" проект. В Docker-Image лежит уже преобученная модель, которую сразу можно использовать для предсказания. https://drive.google.com/file/d/1-fiUokbVUsAO5NLy7G4s-aYme58qrDy_/view?usp=sharing - проект с предобученно  моделью 
