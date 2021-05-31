Запустить FGSM_Attack.py, чтобы получить подпись для картинки, на которую уже была проведена атака. (Для корректной настройки среды раскомментировать первые строчки)
При возникновении ошибок в работе кода обращаться в телеграм - @georgekostenkov .
Для более удобного запуска, демонстрации и установки среды перейти по ссылке ниже в Google Colab Notebook:
https://colab.research.google.com/drive/1H3PkN3vmUgC1S-xOBYEAava8HwgNq91j#scrollTo=wO1vl-Jb-1d_
Возможные ошибки:
-
AttributeError: module 'torch._six' has no attribute 'PY3'
Исправление:
vqa-maskrcnn-benchmark/maskrcnn_benchmark/utils/imports.py - поменять 'PY3' на 'PY37'
-
RuntimeError: Output 0 of SelectBackward is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
Исправление:
vqa-maskrcnn-benchmark/maskrcnn_benchmark/structures/bounding_box.py - закомментировать в функции clip_to_image 4 строки с применением метода 'clamp_' и перезапустить среды выполнения.
