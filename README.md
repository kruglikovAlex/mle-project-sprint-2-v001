# mle-template-case-sprint2

Добро пожаловать в репозиторий-шаблон Практикума для проекта 2 спринта. Ваша цель — улучшить ключевые метрики модели для предсказания стоимости квартир Яндекс Недвижимости.

Проект. Улучшение baseline-модели
Задача - улучшить основную метрику проекта, которая влияет на точность предсказаний стоимости недвижимости и, как следствие, на количество успешных сделок на маркетплейсе.

Цель — сделать процесс воспроизводимым и улучшить ключевые модельные метрики, которые влияют на бизнес-показатели компании, в частности, на увеличение количества успешных сделок.

Имя бакета:  s3-student-mle-20250507-60d03b0a2f-freetrack

Начало работы с проектом:

git clone https://github.com/kruglikovAlex/mle-project-sprint-2-v001.git
pip install -r requirements.txt

тетрадка с исследованием cd model_improvement/project_sprint_2.ipynb
https://github.com/kruglikovAlex/mle-project-sprint-2-v001/blob/9fad29fc7ceb51cec61b66a068e7987750a5c7db/model_improvement/project_sprint_2.ipynb

Этап 1: Разворачивание MLflow и регистрация модели

Для запуска MLFlow необходимо перейти в папку с shell файлом cd mlflow_server
Далее выполнить команду sudo chmod +x ./run_mlflow_registry.sh
Далее запустить сервер командой  ./run_mlflow_registry.sh
EXPERIMENT_NAME = 'base_line_model_CBR_kruglikovAlex'
EXPERIMENT_ID = 3
RUN_NAME = 'base_line_model'
Метрика на базовой модели: rmse = 55956099.8

Этап 2: Проведение EDA

RUN_NAME = 'Feature_engineering'

В ходе исследования данных были проанализированы типы данных, количество пропусков, основные статистики численных признаков и численность в категориальных данных.

Основные выводы:

Признаки studio бесполезен пока = 0 (выборка относительно мала, возможно и будет значение отличное от 0).
Цена жилья растет с увеличением площади и этажности.
Цена на жилье четко делится в цене по году постройки, старый фонд (до 1960 года) имеет более высокую цену.
Сам целевой признак имеет нормальное распределение.
Наиболее широко представленый слелующий тип жилья:
 - c лифтом;
 - тип 4, 2, 1;
 - 9, 17, 12, 5 и 14 этажные строения;
 - основная плотость квартир находится с 1 по 17 этажи;
 - основной сегмет квартир с 1-3 комнатами, причем 1-ых большиство, явно чувствуется что должно стоять отметка "студия"
   второй по плотности блок - 4-5 комнат;
 - 26% общей площадью до 40 м2, 64% - от 40 до 100 м2
 - 44 % квартир жилая площадь 25м2 и менее;
 - 68% квартир с высотой потолков до 3 м.
 - 36 % домов построены более 50 лет назад

Этап 3: Генерация признаков и обучение модели

RUN_ID = 44cc747149c4491590cc02ad0e3759cc
RUN_NAME - 'Feature_engineering'
Полученная метрика с сгенерированными признаками: rmse = 55897738.7
Вручную сгенерированы признаки:

'age': возраст постройки. 
'build_type_floors': тип дома по этажости.
'floor_ratio': соотношение этажей
'kitchen_ratio' и 'living_ratio': соотношение пложадей к общей площади
'building_price_mean' : соотошение цены по зданию
'building_area_mean' : соотношени площади по зданию
'location_cluster' : кластеризация по координатам
'ceiling_height_cluster' : кластеризация по высоте потолка
'flats_count_cluster' : кластеризация по количеству квартир в доме
'dist_origin_dest' : дистанция до центра города
'distance_to_metro_fast' : дистанция до ближайшей станции метро
'floor_group' : '1-й', 'низкий', 'средний', 'высокий', 'очень высокий'
'build_height_group' : 'малоэтажка', 'среднеэтажка', 'высотка', 'небоскреб'
'bad_floor_flag' : флаг "нежелательных" этажей

Автоматически сгенерированы признаки:

Для преобразования числовых признаков использовал энкодеры PolynomialFeatures() и KBinsDiscretizer() для признаков 'floor', 'rooms', 'building_type_int', 'ceiling_height', 'dist_origin_dest', 'distance_to_metro_fast', 'total_area', 'kitchen_area', 'living_area' (если сгенерировать для всех числовых - не хватает ресурсов ВМ). 
и AutoFeatRegressor() с функцией предбразования 'log', больше не позволили ресурсы.

Этап 4: Отбор признаков и обучение новой версии модели

RUN_NAME = 'feature_selected_model'
RUN_ID = 'b4d4a825802b46219d0c4b27e2e1f5e2'

Полученная метрика на отобранном наборе признаков: rmse = 55998188.9
Отбор признаков при помощи SFS двумя способами: 15 признаков с пустого набора и 15 с полного (для сокращения времени 5-ю итарациями по 3 
признака). Далее отобранные признаки были объеденены. Отбор признаков производил на случайной выбрке из 1000 строк тренировочного датасета из-за ограниченного времени работы ВМ (запуск полного датасета никогда не завершался до отключения ВМ).

Итоговоый набор признаков: 
['build_type_floors_low_rise',
 'building_area_mean',
 'building_price_mean',
 'ceiling_height.1',
 'dist_origin_dest',
 'living_area.1',
 'num_fg__KBinsDiscretizer__kitchen_area',
 'num_fg__Polynomial__building_type_int kitchen_area^2',
 'num_fg__Polynomial__ceiling_height total_area',
 'num_fg__Polynomial__ceiling_height total_area kitchen_area',
 'num_fg__Polynomial__ceiling_height total_area^2',
 'num_fg__Polynomial__dist_origin_dest',
 'num_fg__Polynomial__dist_origin_dest distance_to_metro_fast^2',
 'num_fg__Polynomial__dist_origin_dest^2 distance_to_metro_fast',
 'num_fg__Polynomial__dist_origin_dest^2 kitchen_area',
 'num_fg__Polynomial__dist_origin_dest^2 living_area',
 'num_fg__Polynomial__dist_origin_dest^2 total_area',
 'num_fg__Polynomial__floor building_type_int distance_to_metro_fast',
 'num_fg__Polynomial__floor building_type_int^2',
 'num_fg__Polynomial__floor ceiling_height total_area',
 'num_fg__Polynomial__floor total_area^2',
 'num_fg__Polynomial__living_area^2',
 'num_fg__Polynomial__rooms^2',
 'num_fg__Polynomial__total_area living_area^2',
 'num_fg__Polynomial__total_area^2 kitchen_area',
 'num_fg__Polynomial__total_area^2 living_area',
 'total_area.1',
 'total_area_old']

Этап 5: Подбор гиперпараметров и обучение новой версии модели

метод 1:
RUN_NAME = 'RandomizedSearchCV'
RUN_ID = '1c564028330143a9bc994a0b1b4c45c4'

Подбор осуществлялся с помощью RandomizedSearchC:
depth	6
iterations	30
l2_leaf_reg	0.01
learning_rate	0.09999999999999999

Итоговая метрика: rmse = 58912162.4

Метод 2:
RUN_NAME = 'Optuna'
RUN_ID = 8fa43143c91d4ebcb4745fdb1384ae5d

Подбор осуществлялся с помощью Optuna
depth	9
bootstrap_type	Bernoulli
colsample_bylevel	0.9933835132798987
learning_rate	0.06246048591490191

Trial 76
Лучшая метрика: rmse = 46509365.2

Финальная часть - улучшенная модель
RUN_NAME = 'optuna_best_model'
RUN_ID = 016a8bed9c27468c869d774ea10d768b

Финальная модель с отобранными optuna гиперпараметрами: : rmse = 56233814.2
bootstrap_type	Bernoulli
colsample_bylevel	0.9933835132798987
depth	9
eval_metric	RMSE
iterations	300
learning_rate	0.06246048591490191
random_seed	42

В итоге, в общем случае эта модель получилась несколько лучше остальных.