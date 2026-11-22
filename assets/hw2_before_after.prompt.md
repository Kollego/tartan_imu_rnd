# Подписанная схема архитектуры ДЗ 2

Инструмент — встроенный ImageGen. Исходная схема из коммита `953c3f6` использовалась как референс для редактирования.

Redesign the attached Russian before/after network diagram from scratch into a crystal-clear typeset explainer for a first-time reader. It must answer what exactly goes in, what four branches compute and what comes out. Replace the upward velocity arrows and decorative colored cell strips entirely. Use a landscape canvas, white paper, dark navy sans-serif lettering, generous spacing, dark teal accent, flat technical textbook style. Three horizontal sections, all labels directly on objects, large and mobile legible. Only small right-pointing connectors denote computation. No arrows representing spatial direction, no waveform illustrations, no icons for unlabelled quantities.

Top shared input, heading “Вход обеих моделей · 1 секунда IMU”. Below: “200 измерений по 6 чисел”. Two adjacent clearly labelled boxes, one “Акселерометр” with “ax, ay, az · м/с²” below, the other “Гироскоп” with “ωx, ωy, ωz · рад/с” below. These are six channels per time sample.

Middle row heading “Было · MLP”. Three boxes connected left to right: “1200 чисел в один ряд” → “MLP” → output box “Скорость (vx, vy, vz), м/с”. Output as text not an arrow.

Lower row heading “Стало · Conv1d + GRU”. Clearly numbered/signed consecutive computational stages on same row or two lines if required: “4 участка по 50 измерений” → “Conv1d + GRU” → a compact one-column labeled list titled “Четыре прогноза скорости” containing exactly four distinct rows of text “Ветвь 1”, “Ветвь 2”, “Ветвь 3”, “Ветвь 4”. A following block labeled “Gate” with smaller second line “Взвешенная сумма” representing combining all four predictions with weights from encoded input. End with a clear text output box “Скорость (vx, vy, vz), м/с”. Bottom caption “vx, vy, vz — средняя скорость за секунду по осям платформы”. Do not add any other numbers, stray letters, abstract colored squares, unlabeled bars or upward arrows. Professional compact publication layout, no decorative art, no English except MLP, Conv1d, GRU, Gate, IMU and channel symbols. Accurately spell Russian labels.
