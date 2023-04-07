# Credit-Score-Prediction
Предсказание кредитного рейтинга клиентов банка на kaggle: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## Оглавление
[1. Описание проекта] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0)

[2. Цель и задачи проекта] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D1%86%D0%B5%D0%BB%D1%8C-%D0%B8-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0)

[3. Описание данных:] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)

[4. Стадии выполнения проекта:] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D1%81%D1%82%D0%B0%D0%B4%D0%B8%D0%B8-%D0%B2%D1%8B%D0%BF%D0%BE%D0%BB%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0)

[5. Результаты] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D1%80%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%D1%8B)


### Описание проекта
Вы работаете специалистом по обработке данных в глобальной финансовой компании. На протяжении многих лет компания собирала основные банковские реквизиты и собрала много информации, связанной с кредитом. Руководство хочет создать интеллектуальную систему для разделения сотрудников по категориям кредитных баллов, чтобы сократить ручные усилия.

:arrow_up: [к оглавнению] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%B3%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

### Цель и задачи проекта
Цель - предсказание кредитного рейтинга клиентов компании. 

Задачи:
1. Очистка и подготовка датасета к моделированию.
2. Отбор признаков, значимых для построения модели.
3. Учитывая информацию, связанную с кредитом человека, создать модель машинного обучения, которая может классифицировать кредитный рейтинг.

**Метрика**
В качестве метрики качества модели использовалась Accuracy - доля правильных ответов алгоритма.

:arrow_up: [к оглавнению] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%B3%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

### Описание данных:
  0. ID: Уникальный идентификатор записи
  1. Customer_ID: Уникальный идентификатор клиента
  2. Month: Месяц в году
  3. Name: Имя клиента
  4. Age: Возраст клиента
  5. SSN: Номер социального страхования данного лица
  6. Occupation: Род занятий клиента
  7. Annual_Income: Годовой доход лица
  8. Monthly_Inhand_Salary: Ежемесячная заработная плата клиента
  9. Num_Bank_Accounts: Количество банковских счетов данного лица
  10. Num_Credit_Card: Количество кредитных карт, имеющихся у данного лица
  11. Interest_Rate: Процентная ставка по кредитной карте данного лица
  12. Num_of_Loan: Количество кредитов, взятых лицом в банке
  13. Type_of_Loan: Виды кредитов, взятых лицом в банке
  14. Delay_from_due_date: Среднее количество дней, просроченных лицом с момента оплаты
  15. Num_of_Delayed_Payment: Количество платежей, задержанных данным лицом
  16. Changed_Credit_Card: Процентное изменение лимита кредитной карты данного лица
  17. Num_Credit_Inquiries: Количество запросов по кредитной карте, сделанных данным лицом
  18. Credit_Mix: Классификация кредитного портфеля клиента
  19. Outstanding_Debt: Непогашенный баланс клиента
  20. Credit_Utilization_Ratio: Коэффициент использования кредита по кредитной карте клиента
  21. Credit_History_Age: Возраст кредитной истории лица
  22. Payment_of_Min_Amount: Да, если лицо оплатило только минимальную сумму, подлежащую выплате, в противном случае нет
  23. Total_EMI_per_month: Общий EMI человека в месяц
  24. Amount_invested_monthly: Ежемесячная сумма, инвестируемая лицом
  25. Payment_Behaviour: Платежное поведение лица
  26. Monthly_Balance: Ежемесячный остаток, оставшийся на счете данного лица
  27. Credit_Score: Кредитный рейтинг клиента

:arrow_up: [к оглавнению] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%B3%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

### Стадии выполнения проекта:
  1. Предварительный анализ набора данных
  2. Трансформирование и чистка данных
  3. Визуализация
  4. Преобразование данных
  5. Обучение модели
    5.1. Логистическая регрессия
    5.2. Ближайшие соседи
    5.3. Деревья решений
    5.4. Леса рандомизированных деревьев (Рандомный лес)
    5.5 Наивные методы Байеса
    5.6. AdaBoost
    5.7. GridSearchCV

:arrow_up: [к оглавнению] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%B3%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

### Результаты
Результатом является оптимизации гиперпараметров RandomForestClassifier с помощью поиска по сетке GridSearchCV, лучшие результы показала следующая модель:

RandomForestClassifier(max_depth=14, n_estimators=500, random_state=42)

Accuracy на тренировочной выборке: 0.822

Accuracy на тестовой выборке: 0.758

:arrow_up: [к оглавнению] (https://github.com/littleshadowraven/Credit-Score-Prediction/edit/main/README.md#%D0%BE%D0%B3%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)
