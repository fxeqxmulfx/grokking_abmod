// English source messages with Spanish, German, Japanese, French and Russian.
import {helpRows} from './help-translations.ts';
export const languages = ["en","es","de","ja","fr","ru"] as const;
export const rows: readonly (readonly string[])[] = [
  ...helpRows,
  ["Explain Probe contributions", "Explicar las contribuciones al ejemplo", "Beiträge zum Testbeispiel erklären", "テスト式への寄与を説明", "Expliquer les contributions à l’exemple", "Объяснить вклады в ответ"],
  ["Reading Probe contributions","Cómo leer las contribuciones al ejemplo","Beiträge zum Testbeispiel lesen","テスト式への寄与の読み方","Lire les contributions à l’exemple","Как читать график вкладов в ответ"],
  ["One input pair", "Un par de entrada", "Ein Eingabepaar", "1組の入力", "Une paire d’entrée", "Одна пара входов"],
  [
    "The bars describe the current a and b in Probe the model. Changing either input or training the model changes the scores. They are specific to this example.",
    "Las barras describen los valores actuales de a y b en Probar el modelo. Cambiar una entrada o entrenar el modelo cambia los valores. Son específicos de este ejemplo.",
    "Die Balken beziehen sich auf das aktuelle a und b in Modell prüfen. Andere Eingaben oder weiteres Training ändern die Werte. Sie gelten für dieses Beispiel.",
    "棒は「モデルを試す」の現在のaとbに対する値です。入力の変更や学習によって変わり、この例に固有の値です。",
    "Les barres décrivent les valeurs actuelles de a et b dans Tester le modèle. Modifier une entrée ou entraîner le modèle change les scores. Ils sont propres à cet exemple.",
    "Столбцы относятся к текущим a и b в панели «Проверить модель». При смене входов или обучении оценки меняются. Они рассчитаны для этого конкретного примера."
  ],
  [
    "Each channel contributes a vector of 97 logits, one per answer. The score measures the length of that vector after removing its common offset.",
    "Cada canal aporta un vector de 97 logits, uno por respuesta. La puntuación mide la longitud de ese vector tras eliminar su desplazamiento común.",
    "Jeder Kanal trägt einen Vektor mit 97 Logits bei, einen je Antwort. Der Wert misst dessen Länge nach Abzug des gemeinsamen Versatzes.",
    "各チャネルは、97個の答えそれぞれに対応するロジットのベクトルを加えます。スコアは共通のオフセットを除いたベクトルの長さです。",
    "Chaque canal apporte un vecteur de 97 logits, un par réponse. Le score mesure sa longueur après retrait de son décalage commun.",
    "Каждый канал добавляет оценки для 97 возможных ответов. Эти оценки до преобразования в вероятности называются логитами. Вклад канала измеряется длиной вектора оценок после вычитания их общего среднего."
  ],
  ["How the score is computed", "Cómo se calcula la puntuación", "Berechnung des Werts", "スコアの計算方法", "Calcul du score", "Как вычисляется оценка"],
  [
    "h[c] is the channel activation after the final RMSNorm. The centered readout norm measures how strongly its weights distinguish the 97 answers. Their product is the displayed score.",
    "h[c] es la activación del canal tras la RMSNorm final. La norma de los pesos de salida centrados mide cuánto distinguen las 97 respuestas. Su producto es la puntuación mostrada.",
    "h[c] ist die Kanalaktivierung nach der letzten RMSNorm. Die Norm des zentrierten Readouts misst, wie stark seine Gewichte die 97 Antworten unterscheiden. Ihr Produkt ist der angezeigte Wert.",
    "h[c]は最後のRMSNorm後のチャネルの活性値です。中心化した読み出しのノルムは、重みが97個の答えをどの程度区別するかを表します。その積が表示スコアです。",
    "h[c] est l’activation du canal après la RMSNorm finale. La norme de la sortie centrée mesure à quel point ses poids distinguent les 97 réponses. Leur produit est le score affiché.",
    "h[c] — значение канала после последней нормализации RMSNorm. Длина вектора центрированных выходных весов показывает, насколько сильно канал различает 97 ответов. Умножаем модуль h[c] на эту длину и получаем оценку на графике."
  ],
  [
    "The score always uses centered readout weights, regardless of the Center R display setting.",
    "La puntuación siempre usa pesos de salida centrados, independientemente de la opción visual Centrar R.",
    "Der Wert verwendet immer zentrierte Ausgabegewichte, unabhängig von der Anzeigeoption R zentrieren.",
    "表示設定「Rを中心化」に関係なく、スコアには常に中心化した読み出し重みを使います。",
    "Le score utilise toujours les poids de sortie centrés, quel que soit le réglage d’affichage Centrer R.",
    "Оценка всегда рассчитывается по центрированным выходным весам, независимо от переключателя «Центрировать R»."
  ],
  ["Bars and Top 16", "Barras y 16 mayores", "Balken und Top 16", "棒と上位16", "Barres et 16 premiers", "Столбцы и Топ-16"],
  [
    "Channels run from left to right in index order. Taller bars mean larger scores; the tallest currently fills the graph. The vertical scale adjusts, so compare proportions rather than pixel heights across training steps.",
    "Los canales van de izquierda a derecha por índice. Las barras más altas tienen mayor puntuación; la mayor ocupa la altura del gráfico. La escala vertical se ajusta: compara proporciones entre pasos de entrenamiento.",
    "Die Kanäle stehen von links nach rechts in Indexreihenfolge. Höhere Balken bedeuten größere Werte; der höchste füllt das Diagramm. Die vertikale Skala passt sich an: Zwischen Trainingsschritten sind die Anteile vergleichbar.",
    "チャネルは左からインデックス順です。棒が高いほどスコアが大きく、最大の棒がグラフの高さを使います。縦軸は自動調整されるため、学習ステップ間では高さのピクセル数より比率を比較します。",
    "Les canaux suivent l’ordre des indices de gauche à droite. Une barre plus haute signifie un score plus élevé ; la plus haute remplit le graphe. L’échelle verticale s’ajuste : comparez les proportions entre étapes.",
    "Каналы идут слева направо по индексу. Чем выше столбец, тем больше оценка; самый высокий занимает всю высоту графика. Масштаб подстраивается, поэтому между шагами обучения сравнивайте пропорции, а не высоту в пикселях."
  ],
  [
    "Top 16 is the sum of the 16 largest channel scores divided by the sum of all channel scores. For example, 80% means those channels carry 80% of this score, not 80% prediction confidence.",
    "16 mayores es la suma de las 16 puntuaciones más altas dividida por la suma de todas. Por ejemplo, 80% significa que esos canales reúnen el 80% de la puntuación, no una confianza del 80% en la predicción.",
    "Top 16 ist die Summe der 16 größten Kanalwerte geteilt durch die Summe aller Kanalwerte. 80% bedeutet etwa, dass diese Kanäle 80% des Gesamtwerts tragen, keine Vorhersagesicherheit von 80%.",
    "上位16は、スコアが最大の16チャネルの合計を全チャネルの合計で割った割合です。80%なら、その16チャネルが合計スコアの80%を占めます。予測の確信度80%という意味ではありません。",
    "16 premiers est la somme des 16 plus grands scores divisée par la somme de tous les scores. Ainsi, 80% signifie qu’ils portent 80% du score total, pas une confiance de prédiction de 80%.",
    "Топ-16 — сумма 16 наибольших оценок, делённая на сумму оценок всех каналов. Например, 80% означает, что эти каналы дают 80% суммарной оценки, а не уверенность модели 80%."
  ],
  [
    "The score measures magnitude. It does not show which answer a channel favors, whether it helps the correct answer, or the effect of removing it.",
    "La puntuación mide magnitud. No indica qué respuesta favorece un canal, si ayuda a la respuesta correcta ni qué pasaría al eliminarlo.",
    "Der Wert misst den Betrag. Er zeigt weder die vom Kanal bevorzugte Antwort noch, ob er der richtigen Antwort hilft oder was sein Entfernen bewirken würde.",
    "スコアは大きさの指標です。チャネルがどの答えを支持するか、正解に役立つか、取り除くとどうなるかは示しません。",
    "Le score mesure une amplitude. Il n’indique ni la réponse favorisée par un canal, ni s’il aide la bonne réponse, ni l’effet de sa suppression.",
    "Оценка отражает величину вклада. Она не показывает, какой ответ поддерживает канал, помогает ли он правильному ответу и что произойдёт при его удалении."
  ],
  ["Explore a channel", "Explorar un canal", "Einen Kanal untersuchen", "チャネルを調べる", "Explorer un canal", "Исследовать канал"],
  [
    "Tap or click a bar to select its channel. The selected bar is dark; the channel selection is shared with the network and weight maps.",
    "Toca o haz clic en una barra para seleccionar su canal. La barra elegida es oscura; la selección se comparte con la red y los mapas de pesos.",
    "Tippe oder klicke auf einen Balken, um seinen Kanal zu wählen. Der gewählte Balken ist dunkel; die Auswahl gilt auch für Netzwerk und Gewichtskarten.",
    "棒をタップまたはクリックすると、そのチャネルを選択します。選択した棒は暗い色になり、ネットワークや重みマップの選択とも連動します。",
    "Touchez ou cliquez sur une barre pour choisir son canal. La barre sélectionnée est sombre ; le choix est partagé avec le réseau et les cartes de poids.",
    "Нажмите на столбец, чтобы выбрать канал. Выбранный столбец выделяется тёмным цветом; тот же канал выбирается на схеме сети и картах весов."
  ],
  [
    "On a narrow screen, several channels may share a bar. It shows the largest score in that group and selects that channel. Use the Channel field to select any exact index.",
    "En una pantalla estrecha, varios canales pueden compartir una barra. Muestra la mayor puntuación del grupo y selecciona ese canal. Usa el campo Canal para elegir un índice exacto.",
    "Auf schmalen Bildschirmen können sich mehrere Kanäle einen Balken teilen. Er zeigt den größten Wert der Gruppe und wählt diesen Kanal. Im Feld Kanal lässt sich jeder Index genau wählen.",
    "狭い画面では複数のチャネルが1本の棒にまとまることがあります。グループ内の最大スコアを表示し、そのチャネルを選択します。正確なインデックスはチャネル欄で指定できます。",
    "Sur un écran étroit, plusieurs canaux peuvent partager une barre. Elle affiche le plus grand score du groupe et choisit ce canal. Le champ Canal permet de choisir un indice exact.",
    "На узком экране один столбец может объединять несколько каналов. Он показывает наибольшую оценку в группе и выбирает соответствующий канал. Поле «Канал» позволяет указать любой индекс точно."
  ],
  ["Introduction text", "Texto de introducción", "Einführungstext", "導入の本文", "Texte d’introduction", "Текст вступления"],
  ["Reading the weight maps", "Cómo leer los mapas de pesos", "Gewichtskarten lesen", "重みマップの読み方", "Lire les cartes de poids", "Как читать карты весов"],
  ["The three matrices", "Las tres matrices", "Die drei Matrizen", "3つの行列", "Les trois matrices", "Три матрицы"],
  [
    "E: rows are residues 0–96; columns are channels. The same embedding supplies both inputs and the output readout.",
    "E: las filas son residuos 0–96 y las columnas son canales. El mismo embedding sirve para ambas entradas y la salida.",
    "E: Zeilen sind die Reste 0–96, Spalten die Kanäle. Dasselbe Embedding dient beiden Eingaben und der Ausgabe.",
    "Eの行は剰余0〜96、列はチャネルです。同じ埋め込みを2つの入力と出力の読み出しに使います。",
    "E : les lignes sont les résidus 0–96 et les colonnes les canaux. Le même embedding sert aux deux entrées et à la sortie.",
    "E: строки соответствуют остаткам от 0 до 96, столбцы — каналам. Эта матрица переводит числа в значения каналов. Она используется для обоих входов и для выходного преобразования."
  ],
  [
    "A: rows are input channels; columns are output channels. Each cell is one learned connection in the Dense layer.",
    "A: las filas son canales de entrada y las columnas canales de salida. Cada celda es una conexión aprendida de la capa densa.",
    "A: Zeilen sind Eingabekanäle, Spalten Ausgabekanäle. Jede Zelle ist eine gelernte Verbindung der Dense-Schicht.",
    "Aの行は入力チャネル、列は出力チャネルです。各セルは全結合層で学習する1つの接続を表します。",
    "A : les lignes sont les canaux d’entrée et les colonnes les canaux de sortie. Chaque cellule est une connexion apprise de la couche dense.",
    "A: строки — входные каналы, столбцы — выходные. Каждая ячейка — одна обучаемая связь полносвязного слоя."
  ],
  [
    "On the readout map, rows are the 97 answers and columns are channels. E and the readout share weights: they show two roles of one learned matrix.",
    "En el mapa de salida, las filas son las 97 respuestas y las columnas los canales. E y la salida comparten pesos: muestran dos funciones de una sola matriz aprendida.",
    "In der Ausgabekarte sind die Zeilen die 97 Antworten und die Spalten die Kanäle. E und Readout teilen Gewichte: Sie zeigen zwei Rollen derselben gelernten Matrix.",
    "読み出しマップの行は97個の答え、列はチャネルです。Eと読み出しは重みを共有し、同じ学習行列の2つの役割を表示します。",
    "Sur la carte de sortie, les lignes sont les 97 réponses et les colonnes les canaux. E et la sortie partagent leurs poids : deux rôles d’une même matrice apprise.",
    "На карте выходных весов строки соответствуют 97 ответам, столбцы — каналам. Здесь используются те же веса, что и в E: это две роли одной обучаемой матрицы."
  ],
  ["Colors and scale", "Colores y escala", "Farben und Skala", "色とスケール", "Couleurs et échelle", "Цвета и шкала"],
  [
    "Pink means positive, blue means negative; darker colors mean a larger magnitude. Exactly zero has a neutral background.",
    "El rosa indica valores positivos y el azul negativos; un color más oscuro indica mayor magnitud. El cero exacto tiene un fondo neutro.",
    "Rosa steht für positive, Blau für negative Werte. Dunklere Farben bedeuten einen größeren Betrag. Genau null hat einen neutralen Hintergrund.",
    "ピンクは正、青は負の値を示し、色が濃いほど絶対値が大きくなります。ちょうど0のセルは中立色です。",
    "Le rose indique une valeur positive, le bleu une valeur négative ; une couleur plus sombre indique une amplitude plus grande. Zéro a un fond neutre.",
    "Розовый означает положительное значение, синий — отрицательное. Чем темнее цвет, тем больше модуль. Точному нулю соответствует нейтральный фон."
  ],
  [
    "The scale is logarithmic and fixed over time. E and A use separate initial reference scales; the max label shows the largest displayed magnitude.",
    "La escala es logarítmica y permanece fija. E y A usan referencias iniciales distintas; la etiqueta max muestra la mayor magnitud visualizada.",
    "Die Skala ist logarithmisch und bleibt über die Zeit gleich. E und A haben getrennte Anfangsreferenzen; max zeigt den größten dargestellten Betrag.",
    "スケールは対数的で、学習中も固定です。EとAは別々の初期基準値を使います。maxは表示値の絶対値の最大を示します。",
    "L’échelle est logarithmique et reste fixe. E et A ont des références initiales distinctes ; max indique la plus grande amplitude affichée.",
    "Шкала логарифмическая и не меняется со временем. Для E и A используются отдельные начальные масштабы. Подпись «макс» показывает наибольший модуль отображаемых значений."
  ],
  ["Weights and Change ×4", "Pesos y Cambio ×4", "Gewichte und Änderung ×4", "重みと変化×4", "Poids et Variation ×4", "Веса и Изменение ×4"],
  [
    "Weights shows current values. Change ×4 shows the current weight minus its initial value, with a four-times more sensitive color scale.",
    "Pesos muestra los valores actuales. Cambio ×4 muestra el peso actual menos su valor inicial, con una escala de color cuatro veces más sensible.",
    "Gewichte zeigt die aktuellen Werte. Änderung ×4 zeigt aktuelles Gewicht minus Anfangswert, mit einer viermal empfindlicheren Farbskala.",
    "重みは現在の値を表示します。変化×4は現在の重みから初期値を引き、色の感度を4倍にして表示します。",
    "Poids affiche les valeurs actuelles. Variation ×4 affiche le poids actuel moins sa valeur initiale, avec une échelle de couleur quatre fois plus sensible.",
    "Режим «Веса» показывает текущие значения. «Изменение ×4» показывает разность текущего и начального веса с вчетверо более чувствительной цветовой шкалой."
  ],
  [
    "In Change ×4, pink marks an increase and blue a decrease. A neutral cell has no change.",
    "En Cambio ×4, el rosa indica un aumento y el azul una disminución. Una celda neutra no ha cambiado.",
    "In Änderung ×4 bedeutet Rosa eine Zunahme und Blau eine Abnahme. Eine neutrale Zelle hat sich nicht verändert.",
    "変化×4ではピンクが増加、青が減少を示します。中立色のセルは変化していません。",
    "Dans Variation ×4, le rose indique une hausse et le bleu une baisse. Une cellule neutre n’a pas changé.",
    "В режиме «Изменение ×4» розовый означает увеличение, синий — уменьшение. Нейтральная ячейка означает отсутствие изменения."
  ],
  [
    "Hover a cell to see its raw current weight, even in centered or change views. Click a column to select its channel; the border marks that channel across the maps.",
    "Pasa el cursor sobre una celda para ver su peso actual sin transformar, incluso al centrar o mostrar cambios. Haz clic en una columna para seleccionar su canal; el borde lo marca en los mapas.",
    "Beim Überfahren einer Zelle erscheint ihr aktuelles Rohgewicht, auch in zentrierter oder Änderungsansicht. Ein Klick auf eine Spalte wählt ihren Kanal; der Rahmen markiert ihn in allen Karten.",
    "セルにポインターを合わせると、中心化や変化の表示中でも現在の生の重みを確認できます。列をクリックするとチャネルを選択し、各マップで枠が付きます。",
    "Survolez une cellule pour voir son poids actuel brut, même en vue centrée ou de variation. Cliquez sur une colonne pour sélectionner son canal ; le cadre le repère sur les cartes.",
    "Наведите курсор на ячейку, чтобы увидеть её текущий исходный вес, даже при центрировании или просмотре изменений. Нажатие на столбец выбирает канал; рамка отмечает его на всех картах."
  ],
  [
    "In Change ×4, Center R centers the changes across answers.",
    "En Cambio ×4, Centrar R centra los cambios entre respuestas.",
    "In Änderung ×4 zentriert R zentrieren die Änderungen über die Antworten.",
    "変化×4では、Rの中心化が答え全体にわたる変化量の平均を引きます。",
    "Dans Variation ×4, Centrer R centre les variations sur les réponses.",
    "В режиме «Изменение ×4» переключатель «Центр R» центрирует изменения по ответам."
  ],
  ["What does Center R mean?", "¿Qué significa Centrar R?", "Was bedeutet R zentrieren?", "Rの中心化とは？", "Que signifie Centrer R ?", "Что означает «Центрировать R»?"],
  [
    "R means readout: the model's output weights, here Eᵀ.",
    "R significa readout: los pesos de salida del modelo, aquí Eᵀ.",
    "R steht für Readout: die Ausgabegewichte des Modells, hier Eᵀ.",
    "Rはreadout、つまりモデルの出力重みです。このモデルではEᵀです。",
    "R signifie readout : les poids de sortie du modèle, ici Eᵀ.",
    "R — выходные веса модели (от английского readout). В этой архитектуре R = Eᵀ."
  ],
  [
    "For each channel, Center R subtracts the mean weight across all 97 answers. Colors show each answer's deviation from that channel's average weight.",
    "Para cada canal, Centrar R resta el peso medio de las 97 respuestas. Los colores muestran la desviación de cada respuesta respecto al peso medio del canal.",
    "R zentrieren zieht für jeden Kanal den mittleren Wert über alle 97 Antworten ab. Die Farben zeigen die Abweichung jeder Antwort vom mittleren Gewicht dieses Kanals.",
    "Rの中心化は、各チャネルで97個の答えに対する重みの平均を引きます。色は各回答の重みがそのチャネルの平均からどれだけずれるかを示します。",
    "Centrer R soustrait, pour chaque canal, la moyenne des poids sur les 97 réponses. Les couleurs montrent l’écart de chaque réponse par rapport au poids moyen du canal.",
    "«Центрировать R» вычитает для каждого канала средний вес по всем 97 ответам. Цвет показывает, насколько вес конкретного ответа отличается от среднего для этого канала."
  ],
  [
    "This changes only the readout heatmap. Weights, training and predictions stay the same.",
    "Esto solo cambia el mapa de calor de salida. Los pesos, el entrenamiento y las predicciones se mantienen.",
    "Das ändert nur die Heatmap der Ausgabegewichte. Gewichte, Training und Vorhersagen bleiben gleich.",
    "変わるのは出力重みのヒートマップ表示だけです。重み、学習、予測は変わりません。",
    "Seule la carte des poids de sortie change. Les poids, l’entraînement et les prédictions restent identiques.",
    "Меняется только отображение выходных весов на тепловой карте. Веса модели, обучение и предсказания остаются прежними."
  ],
  ["Settings", "Ajustes", "Einstellungen", "設定", "Réglages", "Настройки"],
  ["Custom", "Propio", "Eigene", "カスタム", "Perso", "Свои"],
  ["Custom experiment", "Experimento propio", "Eigenes Experiment", "カスタム実験", "Expérience personnalisée", "Свой эксперимент"],
  ["Learning rate (lr)", "Tasa de aprendizaje (lr)", "Lernrate (lr)", "学習率 (lr)", "Taux d’apprentissage (lr)", "Скорость обучения (lr)"],
  ["Weight decay (wd)", "Decaimiento de pesos (wd)", "Gewichtsabnahme (wd)", "重み減衰 (wd)", "Décroissance des poids (wd)", "Затухание весов (wd)"],
  ["Apply and open training", "Aplicar y abrir entrenamiento", "Anwenden und Training öffnen", "適用して学習を開く", "Appliquer et ouvrir l’entraînement", "Применить и открыть обучение"],
  [
    "Choose the model width and AdamW parameters. Applying starts a fresh custom experiment; presets keep their weights and curves.",
    "Elige la anchura y los parámetros de AdamW. Al aplicar se inicia un nuevo experimento propio; los ajustes predefinidos conservan sus pesos y curvas.",
    "Wähle Modellbreite und AdamW-Parameter. Anwenden beginnt ein neues eigenes Experiment; die Voreinstellungen behalten Gewichte und Kurven.",
    "モデルの幅とAdamWのパラメータを選びます。適用すると新しいカスタム実験を作成します。プリセットの重みと曲線は保持されます。",
    "Choisissez la largeur et les paramètres AdamW. L’application crée une nouvelle expérience personnalisée ; les préréglages conservent leurs poids et courbes.",
    "Выберите ширину модели и параметры AdamW. Применение создаёт новый свой эксперимент; готовые настройки сохраняют веса и графики."
  ],
  [
    "Available widths: 16, 32, 64, 128 and 256.",
    "Anchuras disponibles: 16, 32, 64, 128 y 256.",
    "Verfügbare Breiten: 16, 32, 64, 128 und 256.",
    "選べる幅は16、32、64、128、256です。",
    "Largeurs disponibles : 16, 32, 64, 128 et 256.",
    "Доступные ширины: 16, 32, 64, 128 и 256."
  ],
  ["Choose width 16, 32, 64, 128 or 256.", "Elige una anchura de 16, 32, 64, 128 o 256.", "Wähle eine Breite von 16, 32, 64, 128 oder 256.", "幅は16、32、64、128、256から選んでください。", "Choisissez une largeur de 16, 32, 64, 128 ou 256.", "Выберите ширину 16, 32, 64, 128 или 256."],
  ["Learning rate must be a positive finite number.", "La tasa de aprendizaje debe ser un número finito positivo.", "Die Lernrate muss eine positive endliche Zahl sein.", "学習率は正の有限数にしてください。", "Le taux d’apprentissage doit être un nombre fini positif.", "Скорость обучения должна быть положительным конечным числом."],
  ["Weight decay must be a nonnegative finite number.", "El decaimiento de pesos debe ser un número finito no negativo.", "Die Gewichtsabnahme muss eine nichtnegative endliche Zahl sein.", "重み減衰は0以上の有限数にしてください。", "La décroissance des poids doit être un nombre fini positif ou nul.", "Затухание весов должно быть неотрицательным конечным числом."],
  [
    "Rust/CUDA, seeds {seeds}: epochs to ≥99% held-out accuracy: {epochs}. A dash means not reached within {budget} epochs. Browser crossings may differ.",
    "Rust/CUDA, semillas {seeds}: épocas hasta ≥99% en prueba: {epochs}. Un guion indica que no se alcanzó en {budget} épocas. En el navegador puede variar.",
    "Rust/CUDA, Seeds {seeds}: Epochen bis ≥99% Testgenauigkeit: {epochs}. Ein Strich bedeutet: in {budget} Epochen nicht erreicht. Browser-Schwellen können abweichen.",
    "Rust/CUDA、シード{seeds}：テスト99%以上までのエポック数は{epochs}。ダッシュは{budget}エポック以内に未達を示します。ブラウザーでは到達時期が異なる場合があります。",
    "Rust/CUDA, graines {seeds} : époques jusqu’à ≥99% au test : {epochs}. Un tiret signifie non atteint en {budget} époques. Le navigateur peut différer.",
    "Контрольные запуски Rust/CUDA с seed {seeds}: число эпох до точности теста не ниже 99% — {epochs}. Прочерк означает, что порог не достигнут за {budget} эпох. В браузере результат может отличаться."
  ],
  ["Start", "Inicio", "Start", "導入", "Début", "Начало"],
  ["Idea", "Idea", "Idee", "狙い", "Idée", "Идея"],
  ["Epochs", "Épocas", "Epochen", "エポック", "Époques", "Эпохи"],
  ["Paper", "Artículo", "Artikel", "論文", "Article", "Статья"],
  ["Chart", "Gráfico", "Kurven", "グラフ", "Courbes", "График"],
  ["Grokking", "Grokking", "Grokking", "Grokking", "Grokking", "Grokking"],
  ["Findings", "Hallazgos", "Befunde", "観察結果", "Constats", "Наблюдения"],
  ["Introduction pages", "Páginas de introducción", "Einführungsseiten", "導入ページ", "Pages d’introduction", "Страницы вступления"],
  ["Explain curves and plateau shading", "Explicar las curvas y las zonas de meseta", "Kurven und Plateauflächen erklären", "曲線と停滞期の色分けを説明", "Expliquer les courbes et les plateaux", "Объяснение кривых и выделенных плато"],
  ["Learning after memorization", "Aprender después de memorizar", "Lernen nach dem Auswendiglernen", "暗記した後の学習", "Apprendre après avoir mémorisé", "Обучение после запоминания"],
  [
    "In 2022, OpenAI researchers trained small Transformers on arithmetic tables such as division modulo 97. The networks memorized the training examples early but solved unseen pairs only much later. They called this late generalization grokking.",
    "En 2022, investigadores de OpenAI entrenaron Transformers pequeños con tablas aritméticas como la división módulo 97. Las redes memorizaban pronto los ejemplos de entrenamiento, pero resolvían pares nuevos mucho después. Llamaron grokking a esta generalización tardía.",
    "2022 trainierten Forschende von OpenAI kleine Transformer auf Rechentabellen wie der Division modulo 97. Die Netze lernten die Trainingsbeispiele früh auswendig, lösten neue Paare aber erst viel später. Diese späte Verallgemeinerung nannten sie Grokking.",
    "2022年、OpenAIの研究者たちは、97を法とする除算などの演算表で小さなTransformerを学習させました。ネットワークは学習例を早く暗記しましたが、未知の組を解けるようになったのはずっと後でした。この遅れて起こる汎化はgrokkingと名付けられました。",
    "En 2022, des chercheurs d’OpenAI ont entraîné de petits Transformers sur des tables arithmétiques comme la division modulo 97. Les réseaux mémorisaient vite les exemples d’entraînement, mais ne résolvaient les paires nouvelles que bien plus tard. Ils ont appelé grokking cette généralisation tardive.",
    "В 2022 году исследователи OpenAI обучали небольшие трансформеры на арифметических таблицах, например на делении по модулю 97. Сети быстро запоминали обучающие примеры, но новые пары начинали решать лишь намного позже. Это позднее обобщение назвали grokking."
  ],
  [
    "This lab is an attempt to find out in detail how it works. Train the same task in your browser, change width, learning rate (lr), weight decay (wd) and seed, and follow accuracy, loss and every weight epoch by epoch.",
    "Este laboratorio intenta averiguar en detalle cómo funciona. Entrena la misma tarea en el navegador, cambia la anchura, la tasa de aprendizaje (lr), el decaimiento de pesos (wd) y la semilla, y sigue la precisión, la pérdida y cada peso época a época.",
    "Dieses Labor versucht herauszufinden, wie das im Detail funktioniert. Trainiere dieselbe Aufgabe im Browser, ändere Breite, Lernrate (lr), Weight Decay (wd) und Seed und verfolge Genauigkeit, Loss und jedes Gewicht Epoche für Epoche.",
    "この実験は、その仕組みを詳しく調べる試みです。同じ課題をブラウザーで学習させ、幅、学習率（lr）、重み減衰（wd）、シードを変えながら、正解率、損失、すべての重みをエポックごとに追えます。",
    "Ce laboratoire tente de comprendre en détail comment cela fonctionne. Entraînez la même tâche dans le navigateur, changez la largeur, le taux d’apprentissage (lr), la décroissance des poids (wd) et la graine, et suivez exactitude, perte et chaque poids, époque après époque.",
    "Эта лаборатория помогает подробно разобраться в таком обучении. Обучайте модель в браузере, меняйте ширину, скорость обучения (lr), затухание весов (wd) и seed — число для начальных условий. Следите за точностью, ошибкой и изменением весов от эпохи к эпохе."
  ],
  [
    "The Findings tab summarizes what the experiments have shown so far.",
    "La pestaña Hallazgos resume lo que han mostrado los experimentos hasta ahora.",
    "Der Reiter Befunde fasst zusammen, was die Experimente bisher zeigen.",
    "これまでの実験でわかったことは「観察結果」にまとめています。",
    "L’onglet Constats résume ce que les expériences ont montré jusqu’ici.",
    "Что уже показали эксперименты — во вкладке «Наблюдения»."
  ],
  ["The 2022 paper", "El artículo de 2022", "Die Arbeit von 2022", "2022年の論文", "L’article de 2022", "Статья 2022 года"],
  [
    "The paper used small Transformers. In its division-modulo-97 example, training accuracy was nearly perfect before 1,000 optimizer steps; validation approached that level near 1,000,000 steps.",
    "El artículo usó Transformers pequeños. En el ejemplo de división módulo 97, el entrenamiento fue casi perfecto antes de 1.000 pasos; la validación llegó a ese nivel cerca de 1.000.000 de pasos.",
    "Die Arbeit nutzte kleine Transformer. Beim Teilen modulo 97 war die Trainingsgenauigkeit vor 1.000 Schritten fast perfekt; die Validierung erreichte diesen Bereich erst nahe 1.000.000 Schritten.",
    "論文では小さなTransformerを使いました。97を法とする除算の例では、学習正解率は1,000更新前にほぼ完全になり、検証正解率は約1,000,000更新で同じ水準に達しました。",
    "L’article utilisait de petits Transformers. Pour la division modulo 97, l’entraînement était presque parfait avant 1 000 mises à jour ; la validation approchait ce niveau vers 1 000 000.",
    "Авторы использовали небольшие трансформеры. В примере деления по модулю 97 train стал почти идеальным до 1 000 шагов оптимизатора, а validation достиг сопоставимого уровня около 1 000 000 шагов."
  ],
  [
    "This lab uses a Dense model for the same arithmetic task. The paper reports optimizer steps; this lab counts complete epochs.",
    "Aquí una red densa aprende la misma tarea. El artículo cuenta pasos del optimizador; este laboratorio cuenta épocas completas.",
    "Hier lernt ein dichtes Modell dieselbe Aufgabe. Die Arbeit zählt Optimiererschritte; dieses Labor zählt vollständige Epochen.",
    "ここでは全結合モデルが同じ算術課題を学びます。論文は更新回数、この実験は完全なエポック数を表示します。",
    "Ici, un modèle dense apprend la même tâche. L’article compte les mises à jour ; ce laboratoire compte les époques complètes.",
    "Здесь ту же арифметическую задачу изучает плотная модель. В статье считают шаги оптимизатора; здесь — полные эпохи."
  ],
  ["What is an epoch?", "¿Qué es una época?", "Was ist eine Epoche?", "エポックとは？", "Qu’est-ce qu’une époque ?", "Что такое эпоха?"],
  [
    "One epoch uses all 4,656 training equations once, in shuffled order.",
    "Una época recorre una vez las 4.656 ecuaciones de entrenamiento, en orden aleatorio.",
    "Eine Epoche nutzt alle 4.656 Trainingsgleichungen genau einmal in gemischter Reihenfolge.",
    "1エポックで4,656個の学習例をシャッフルした順に1回ずつ使います。",
    "Une époque utilise les 4 656 équations d’entraînement une fois, dans un ordre mélangé.",
    "За одну эпоху все 4 656 обучающих уравнений используются по одному разу, в перемешанном порядке."
  ],
  [
    "One optimizer step updates weights from one batch. Each epoch contains ten steps.",
    "Un paso del optimizador actualiza los pesos con un lote. Cada época contiene diez pasos.",
    "Ein Optimiererschritt aktualisiert Gewichte anhand eines Batches. Jede Epoche umfasst zehn Schritte.",
    "1回の更新で1バッチから重みを更新します。1エポックには10回の更新があります。",
    "Une mise à jour ajuste les poids à partir d’un lot. Chaque époque comprend dix mises à jour.",
    "Один шаг оптимизатора обновляет веса по одному батчу. В каждой эпохе десять шагов."
  ],
  [
    "Test pairs are evaluated after every epoch and never update weights.",
    "Los pares de prueba se evalúan tras cada época y nunca actualizan los pesos.",
    "Testpaare werden nach jeder Epoche geprüft und aktualisieren niemals die Gewichte.",
    "テストの組は各エポック後に評価し、重みの更新には使いません。",
    "Les paires de test sont évaluées après chaque époque sans modifier les poids.",
    "Тестовые пары проверяются после каждой эпохи и не участвуют в обновлении весов."
  ],
  ["1 epoch", "1 época", "1 Epoche", "1エポック", "1 époque", "1 эпоха"],
  ["10 AdamW updates per epoch", "10 pasos AdamW por época", "10 AdamW-Schritte pro Epoche", "1エポックでAdamWを10回更新", "10 mises à jour AdamW par époque", "10 обновлений AdamW за эпоху"],
  [
    "Loss measures prediction error; lower is better. Accuracy is the fraction of correct answers; higher is better.",
    "La pérdida mide el error: cuanto menor, mejor. La precisión es la proporción de respuestas correctas: cuanto mayor, mejor.",
    "Loss misst den Vorhersagefehler: kleiner ist besser. Genauigkeit ist der Anteil richtiger Antworten: größer ist besser.",
    "損失は予測の誤差で、低いほど良好です。正解率は正答の割合で、高いほど良好です。",
    "La perte mesure l’erreur : plus faible est mieux. L’exactitude est la proportion de bonnes réponses : plus élevée est mieux.",
    "Ошибка (loss) оценивает качество предсказаний: чем она меньше, тем лучше. Точность — доля правильных ответов: чем она выше, тем лучше."
  ],
  ["Reading the learning curves", "Leer las curvas de aprendizaje", "Lernkurven lesen", "学習曲線の見方", "Lire les courbes d’apprentissage", "Как читать график обучения"],
  [
    "Orange is training; blue is held-out test data. Compare the curves to see whether the model learns unseen pairs.",
    "Naranja es entrenamiento; azul son datos de prueba. Compara las curvas para ver si el modelo aprende pares nuevos.",
    "Orange steht für Training, Blau für zurückgehaltene Testdaten. Vergleiche die Kurven, um das Lernen neuer Paare zu erkennen.",
    "オレンジは学習、青は未学習のテストデータです。曲線を比べると未知の組への汎化が分かります。",
    "L’orange représente l’entraînement, le bleu le test. Comparez les courbes pour voir si le modèle apprend les paires nouvelles.",
    "Оранжевая кривая показывает результат на обучающих парах, синяя — на тестовых. Сравнивайте их, чтобы понять, научилась ли модель решать новые пары."
  ],
  [
    "Beige marks the longest continuous interval with train ≥99% and test ≤10%, before the first test ≥90% crossing.",
    "El beige marca el intervalo continuo más largo con entrenamiento ≥99% y prueba ≤10%, antes del primer resultado de prueba ≥90%.",
    "Beige markiert das längste durchgehende Intervall mit Training ≥99% und Test ≤10%, bevor der Test erstmals ≥90% erreicht.",
    "ベージュは、テストが初めて90%以上になる前の、学習99%以上・テスト10%以下が最長で続く区間です。",
    "Le beige marque le plus long intervalle continu avec entraînement ≥99% et test ≤10%, avant le premier test ≥90%.",
    "Бежевым выделен самый длинный непрерывный период, когда точность обучения не ниже 99%, а точность теста — не выше 10%. Учитывается время до первого достижения 90% на тесте."
  ],
  [
    "Violet marks the same conditions with exactly 100% train accuracy.",
    "El violeta marca las mismas condiciones con entrenamiento exactamente al 100%.",
    "Violett markiert dieselben Bedingungen mit exakt 100% Trainingsgenauigkeit.",
    "紫は同じ条件で学習正解率がちょうど100%の区間です。",
    "Le violet marque les mêmes conditions avec exactement 100% à l’entraînement.",
    "Фиолетовый показывает такой же период, но с точностью обучения ровно 100%."
  ],
  [
    "Shading describes accuracy conditions, even on the loss view. It does not mean the loss or weights stopped changing.",
    "El sombreado indica condiciones de precisión, incluso en la vista de pérdida. No significa que la pérdida o los pesos dejen de cambiar.",
    "Die Flächen beschreiben Genauigkeitsbedingungen, auch in der Loss-Ansicht. Loss und Gewichte können sich dabei weiter ändern.",
    "損失の表示中も色分けは正解率の条件を示します。損失や重みの変化が止まったという意味ではありません。",
    "Les zones décrivent l’exactitude, même dans la vue des pertes. Elles ne signifient pas que les pertes ou les poids ne changent plus.",
    "Выделенные области показывают условия по точности, даже когда открыт график ошибки. Они не означают, что ошибка или веса перестали меняться."
  ],
  ["When does it count as grokking?", "¿Cuándo cuenta como grokking?", "Wann zählt es als Grokking?", "Grokkingの判定条件", "Quand parle-t-on de grokking ?", "Когда это считается grokking?"],
  [
    "A shaded interval shows memorization with poor generalization. It does not by itself establish a later jump.",
    "Un intervalo sombreado muestra memorización con poca generalización. Por sí solo no demuestra un salto posterior.",
    "Ein markiertes Intervall zeigt Auswendiglernen bei schwacher Verallgemeinerung. Es allein belegt keinen späteren Sprung.",
    "色の区間は、暗記しても汎化が弱い状態です。その区間だけでは後の急上昇を証明しません。",
    "Une zone indique la mémorisation sans bonne généralisation. Elle ne prouve pas à elle seule un bond ultérieur.",
    "Выделенный интервал показывает запоминание при слабом обобщении. Сам по себе он ещё не доказывает последующий скачок."
  ],
  [
    "This lab requires at least 25 plateau epochs, test ≤10% → ≥90% within 250 epochs, and a later test ≥99% result. These are this lab's thresholds, not a definition imposed by the paper.",
    "Este laboratorio exige al menos 25 épocas de meseta, prueba ≤10% → ≥90% en 250 épocas y después prueba ≥99%. Son umbrales del laboratorio, no una definición del artículo.",
    "Dieses Labor verlangt mindestens 25 Plateau-Epochen, Test ≤10% → ≥90% innerhalb von 250 Epochen und später Test ≥99%. Das sind Laborschwellen, keine Definition der Arbeit.",
    "この実験では25エポック以上の停滞、250エポック以内のテスト10%以下→90%以上、後の99%以上を条件にします。論文の定義ではなく、この実験の判定基準です。",
    "Ce laboratoire exige au moins 25 époques de plateau, un test ≤10% → ≥90% en 250 époques, puis ≥99%. Ce sont ses propres seuils, pas une définition imposée par l’article.",
    "Здесь нужны минимум 25 эпох плато, рост test ≤10% → ≥90% за 250 эпох и затем test ≥99%. Эти пороги приняты в лаборатории, а не заданы статьёй."
  ],
  [
    "In Rust/CUDA runs, W16–W64 never held a 25-epoch plateau. W128 held one when weight decay was weak, but test accuracy then rose over hundreds of epochs. Only W256 met all three conditions, at some settings.",
    "En las ejecuciones Rust/CUDA, W16–W64 nunca mantuvieron una meseta de 25 épocas. W128 la mantuvo con un decaimiento de pesos débil, pero después la precisión de prueba subió durante cientos de épocas. Solo W256 cumplió las tres condiciones, con algunos ajustes.",
    "In den Rust/CUDA-Läufen hatten W16–W64 nie ein Plateau von 25 Epochen. W128 hatte eines bei schwachem Weight Decay, doch die Testgenauigkeit stieg danach über Hunderte Epochen. Nur W256 erfüllte alle drei Bedingungen, bei einigen Einstellungen.",
    "Rust/CUDAでの実行では、W16〜W64に25エポックの停滞期は一度も現れませんでした。W128は重み減衰が弱いと停滞期を示しましたが、その後のテスト正解率の上昇には数百エポックかかりました。3つの条件をすべて満たしたのは、一部の設定のW256だけです。",
    "Dans les essais Rust/CUDA, W16–W64 n’ont jamais tenu un plateau de 25 époques. W128 en a tenu un avec une faible décroissance des poids, mais l’exactitude au test a ensuite monté sur des centaines d’époques. Seul W256 a rempli les trois conditions, pour certains réglages.",
    "В прогонах Rust/CUDA у W16–W64 ни разу не было плато в 25 эпох. У W128 оно появлялось при слабом weight decay, но затем точность на тесте росла сотни эпох. Все три условия выполнила только W256, при некоторых настройках."
  ],
  ["lr and wd at W128", "lr y wd en W128", "lr und wd bei W128", "W128でのlrとwd", "lr et wd à W128", "lr и wd на W128"],
  [
    "Up to lr=0.003, the learning rate only sets the time scale: halving it roughly doubles every epoch count.",
    "Hasta lr=0.003, la tasa de aprendizaje solo fija la escala de tiempo: al reducirla a la mitad, cada número de épocas casi se duplica.",
    "Bis lr=0.003 bestimmt die Lernrate nur die Zeitskala: Halbiert man sie, verdoppelt sich jede Epochenzahl ungefähr.",
    "lr=0.003までは、学習率は時間の尺度を決めるだけです。半分にすると、どのエポック数もほぼ2倍になります。",
    "Jusqu’à lr=0.003, le taux d’apprentissage ne fixe que l’échelle de temps : le diviser par deux double à peu près chaque nombre d’époques.",
    "До lr=0.003 скорость обучения задаёт лишь масштаб времени: уменьшите её вдвое — и все сроки примерно удвоятся."
  ],
  [
    "Weight decay sets how far the test lags behind training: at lr=0.001, test reaches 99% after about 240 epochs with wd=1, 400 with wd=0.3, 600 with wd=0.1 and 860–1,290 with wd ≤ 0.01.",
    "El decaimiento de pesos fija cuánto se retrasa la prueba: con lr=0.001, llega al 99% tras unas 240 épocas con wd=1, 400 con wd=0.3, 600 con wd=0.1 y 860–1.290 con wd ≤ 0.01.",
    "Weight Decay bestimmt, wie weit der Test dem Training hinterherläuft: Bei lr=0.001 erreicht er 99% nach etwa 240 Epochen mit wd=1, 400 mit wd=0.3, 600 mit wd=0.1 und 860–1.290 mit wd ≤ 0.01.",
    "重み減衰は、テストが学習にどれだけ遅れるかを決めます。lr=0.001では、テストが99%に達するのはwd=1で約240エポック、wd=0.3で400、wd=0.1で600、wd ≤ 0.01で860〜1,290エポックです。",
    "La décroissance des poids fixe le retard du test sur l’entraînement : à lr=0.001, il atteint 99% après environ 240 époques avec wd=1, 400 avec wd=0.3, 600 avec wd=0.1 et 860–1 290 avec wd ≤ 0.01.",
    "Weight decay задаёт отставание теста от обучения: при lr=0.001 тест достигает 99% примерно за 240 эпох при wd=1, 400 при wd=0.3, 600 при wd=0.1 и 860–1 290 при wd ≤ 0.01."
  ],
  [
    "Too large steps break training: every setting with lr × wd ≥ 0.1 failed, and lr=1 is unstable. Fastest: lr=0.01, wd=1, 99% in 30–32 epochs (the Fast preset).",
    "Los pasos demasiado grandes rompen el entrenamiento: todos los ajustes con lr × wd ≥ 0.1 fallaron, y lr=1 es inestable. Lo más rápido: lr=0.01, wd=1, 99% en 30–32 épocas (el ajuste Rápido).",
    "Zu große Schritte zerstören das Training: Jede Einstellung mit lr × wd ≥ 0.1 schlug fehl, und lr=1 ist instabil. Am schnellsten: lr=0.01, wd=1, 99% nach 30–32 Epochen (Voreinstellung Schnell).",
    "大きすぎるステップは学習を壊します。lr × wd ≥ 0.1の設定はすべて失敗し、lr=1は不安定です。最速はlr=0.01、wd=1で、30〜32エポックで99%（「高速」設定）。",
    "Des pas trop grands cassent l’entraînement : tous les réglages avec lr × wd ≥ 0.1 ont échoué, et lr=1 est instable. Le plus rapide : lr=0.01, wd=1, 99% en 30–32 époques (le réglage Rapide).",
    "Большие шаги ломают обучение: провалились все настройки с lr × wd ≥ 0.1, а lr=1 нестабилен. Быстрее всех lr=0.01, wd=1: 99% за 30–32 эпохи (режим «Быстрый»)."
  ],
  [
    "Rust/CUDA: lr 0.00003–1 × wd 0–1, seeds 11/17/23, up to 10,000 epochs.",
    "Rust/CUDA: lr 0.00003–1 × wd 0–1, semillas 11/17/23, hasta 10.000 épocas.",
    "Rust/CUDA: lr 0.00003–1 × wd 0–1, Seeds 11/17/23, bis zu 10.000 Epochen.",
    "Rust/CUDA：lr 0.00003〜1 × wd 0〜1、シード11/17/23、最大10,000エポック。",
    "Rust/CUDA : lr 0.00003–1 × wd 0–1, graines 11/17/23, jusqu’à 10 000 époques.",
    "Rust/CUDA: lr 0.00003–1 × wd 0–1, seed 11/17/23, до 10 000 эпох."
  ],
  ["Continue training", "Continuar entrenamiento", "Training fortsetzen", "学習を続ける", "Continuer l’entraînement", "Продолжить обучение"],
  ["Paused at 99% held-out accuracy", "Pausa al 99% de precisión de prueba", "Pause bei 99% Testgenauigkeit", "テスト正解率99%で一時停止", "Pause à 99% d’exactitude de test", "Пауза при 99% на тесте"],
  ["Pauses at 99% test accuracy. Continue to train further.", "Pausa al 99% en prueba. Puedes continuar.", "Pause bei 99% im Test. Fortsetzen ist möglich.", "テスト正解率99%で一時停止。学習を続けられます。", "Pause à 99% au test. Vous pouvez continuer.", "Пауза при 99% на тесте. Можно продолжить обучение."],
  ["PyTorch .py", "PyTorch .py", "PyTorch .py", "PyTorch .py", "PyTorch .py", "PyTorch .py"],
  ["Export PyTorch source; run with uv", "Exportar código PyTorch; ejecutar con uv", "PyTorch-Code exportieren; mit uv starten", "PyTorchコードを出力し、uvで実行", "Exporter le code PyTorch ; lancer avec uv", "Экспорт кода PyTorch; запуск через uv"],
  ["Grokking: original paper ↗", "Grokking: artículo original ↗", "Grokking: Originalarbeit ↗", "Grokking：原論文 ↗", "Grokking : article original ↗", "Grokking: исходная статья ↗"],
  [
    "Export PyTorch source to reproduce this architecture and train from the selected seed with uv run. The file contains training code and data generation.",
    "Exporta el código PyTorch para reproducir esta arquitectura y entrenar desde la semilla elegida con uv run. El archivo incluye entrenamiento y generación de datos.",
    "Exportiere PyTorch-Code, um diese Architektur mit dem gewählten Seed über uv run zu trainieren. Die Datei enthält Training und Datengenerierung.",
    "PyTorchコードを出力して、この構造を選択したシードからuv runで学習できます。学習コードとデータ生成が含まれます。",
    "Exportez le code PyTorch pour entraîner cette architecture avec la graine choisie via uv run. Le fichier contient l’entraînement et la génération des données.",
    "Экспортируйте код PyTorch для обучения этой архитектуры с выбранным seed через uv run. Файл содержит обучение и генерацию данных."
  ],
  [
    "Compare one shared-embedding Rotate90 Dense architecture at widths 16, 32, 64, 128 and 256. Train locally in your browser with live loss, accuracy and weights.",
    "Compara una arquitectura Rotate90 densa con embedding compartido y anchuras 16, 32, 64, 128 y 256. Entrena en tu navegador con pérdida, precisión y pesos en directo.",
    "Vergleiche eine dichte Rotate90-Architektur mit gemeinsamem Embedding in den Breiten 16, 32, 64, 128 und 256. Trainiere im Browser mit Verlust, Genauigkeit und Gewichten in Echtzeit.",
    "共有埋め込みを使う全結合Rotate90を幅16、32、64、128、256で比較。ブラウザー内で学習し、損失・正解率・重みをリアルタイムに確認できます。",
    "Comparez une architecture Rotate90 dense à embedding partagé, de largeurs 16, 32, 64, 128 et 256. Entraînez-la dans votre navigateur avec pertes, exactitude et poids en direct.",
    "Сравните одну полносвязную архитектуру Rotate90 с общими эмбеддингами при ширине 16, 32, 64, 128 и 256 каналов. Обучайте её в браузере и наблюдайте за ошибкой, точностью и весами."
  ],
  ["One architecture, five widths", "Una arquitectura, cinco anchuras", "Eine Architektur, fünf Breiten", "同じ構造、5つの幅", "Une architecture, cinq largeurs", "Одна архитектура, пять ширин"],
  [
    "Widths 16 / 32 / 64 / 128 / 256 share one architecture (1,808 / 4,128 / 10,304 / 28,800 / 90,368 parameters). On every Rust/CUDA seed, Fast reaches 99% held-out accuracy earliest and Slow latest. Plateau memorizes longest before it; where Slow does, Plateau has the longest plateau followed by grokking. Only widths from 128 have a memorization plateau. Each width and preset preserves its weights, optimizer and curves.",
    "Las anchuras 16 / 32 / 64 / 128 / 256 comparten arquitectura (1.808 / 4.128 / 10.304 / 28.800 / 90.368 parámetros). En todas las semillas de Rust/CUDA, Rápido alcanza el 99% en prueba antes y Lento más tarde. Meseta memoriza más tiempo antes de lograrlo; si eso le toca a Lento, Meseta tiene la meseta más larga seguida de grokking. Solo las anchuras desde 128 tienen meseta de memorización. Cada anchura y ajuste conserva pesos, optimizador y curvas.",
    "Breiten 16 / 32 / 64 / 128 / 256 teilen eine Architektur (1.808 / 4.128 / 10.304 / 28.800 / 90.368 Parameter). Bei jedem Rust/CUDA-Seed erreicht Schnell 99% Testgenauigkeit am frühesten und Langsam am spätesten. Plateau lernt davor am längsten auswendig; tut das Langsam, hat Plateau das längste Plateau mit anschließendem Grokking. Nur Breiten ab 128 haben ein Auswendiglernplateau. Jede Breite und Voreinstellung behält Gewichte, Optimierer und Kurven.",
    "幅16 / 32 / 64 / 128 / 256は同じ構造です（1,808 / 4,128 / 10,304 / 28,800 / 90,368パラメータ）。Rust/CUDAのどのシードでも、高速はテスト99%に最も早く、低速は最も遅く到達します。停滞期はその前の暗記が最も長く、それが低速の場合は、grokkingが続く停滞期のうち最も長いものです。暗記停滞期があるのは幅128以上だけです。幅と設定ごとに重み・最適化状態・曲線を保持します。",
    "Les largeurs 16 / 32 / 64 / 128 / 256 partagent une architecture (1 808 / 4 128 / 10 304 / 28 800 / 90 368 paramètres). Sur chaque graine Rust/CUDA, Rapide atteint 99% au test le plus tôt et Lent le plus tard. Plateau mémorise le plus longtemps avant ; si c’est le cas de Lent, Plateau a le plus long plateau suivi d’un grokking. Seules les largeurs à partir de 128 ont un plateau de mémorisation. Chaque largeur et réglage conserve poids, optimiseur et courbes.",
    "Ширины 16 / 32 / 64 / 128 / 256 используют одну архитектуру (1 808 / 4 128 / 10 304 / 28 800 / 90 368 параметров). На каждом seed в Rust/CUDA «Быстрый» раньше всех достигает 99% на тесте, а «Медленный» — позже всех. «Плато» дольше всех запоминает перед этим; если это «Медленный», у «Плато» самое долгое плато, за которым следует grokking. Плато запоминания есть только у ширин от 128. Каждая ширина и режим сохраняют веса, оптимизатор и графики."
  ],
  [
    "Seed 11, Rust/CUDA: after 10,000 epochs, train 32.1% and held-out 18.1%; neither reached 99%. Only one seed verified; browser results may differ.",
    "Semilla 11, Rust/CUDA: tras 10.000 épocas, entrenamiento 32,1% y prueba 18,1%; ninguno llegó al 99%. Solo una semilla verificada; el navegador puede variar.",
    "Seed 11, Rust/CUDA: nach 10.000 Epochen Training 32,1%, Test 18,1%; beide unter 99%. Nur ein Seed geprüft; Browser-Ergebnisse können abweichen.",
    "シード11、Rust/CUDA：10,000エポック後、学習32.1%、テスト18.1%。99%には到達しませんでした。検証済みシードは1つで、ブラウザーでは異なる場合があります。",
    "Graine 11, Rust/CUDA : après 10 000 époques, entraînement 32,1%, test 18,1% ; aucun à 99%. Une seule graine vérifiée ; le navigateur peut différer.",
    "Seed 11, Rust/CUDA: после 10 000 эпох train 32,1%, тест 18,1%; 99% не достигнуты. Проверен один seed; результаты в браузере могут отличаться."
  ],
  [
    "Seed 11, Rust/CUDA: train ≥99% at epoch 931; held-out ≥99% at 3690. Only one seed verified; browser crossings may differ.",
    "Semilla 11, Rust/CUDA: entrenamiento ≥99% en la época 931; prueba ≥99% en 3690. Solo una semilla verificada; los umbrales pueden variar en el navegador.",
    "Seed 11, Rust/CUDA: Training ≥99% in Epoche 931; Test ≥99% bei 3690. Nur ein Seed geprüft; Browser-Schwellen können abweichen.",
    "シード11、Rust/CUDA：学習99%以上が931エポック、テスト99%以上が3690。検証済みシードは1つで、ブラウザーでは到達時期が異なる場合があります。",
    "Graine 11, Rust/CUDA : entraînement ≥99% à l’époque 931 ; test ≥99% à 3690. Une seule graine vérifiée ; les seuils peuvent différer dans le navigateur.",
    "Seed 11, Rust/CUDA: train ≥99% на эпохе 931; тест ≥99% на 3690. Проверен один seed; в браузере моменты пересечения порогов могут отличаться."
  ],
  [
    "A[{input},{output}] · {input} → {output} · weight {value}",
    "A[{input},{output}] · {input} → {output} · peso {value}",
    "A[{input},{output}] · {input} → {output} · Gewicht {value}",
    "A[{input},{output}] · {input} → {output} · 重み {value}",
    "A[{input},{output}] · {input} → {output} · poids {value}",
    "A[{input},{output}] · {input} → {output} · вес {value}"
  ],
  [
    "Dense residual block 1, channel {channel}, {width} to 1 neurons, {width} learned edges, pre-RMSNorm, ReLU and one skip connection",
    "Bloque residual denso 1, canal {channel}, {width} a 1 neuronas, {width} conexiones aprendidas, pre-RMSNorm, ReLU y una conexión residual",
    "Dichter Residualblock 1, Kanal {channel}, {width} zu 1 Neuronen, {width} gelernte Verbindungen, Pre-RMSNorm, ReLU und ein Residualpfad",
    "全結合残差ブロック1、チャネル{channel}、{width}対1のニューロン、{width}個の学習する結合、事前RMSNorm、ReLUと残差経路1つ",
    "Bloc résiduel dense 1, canal {channel}, {width} vers 1 neurones, {width} connexions apprises, pre-RMSNorm, ReLU et un chemin résiduel",
    "Плотный residual-блок 1, канал {channel}, {width} к 1 нейронов, {width} обучаемых связей, pre-RMSNorm, ReLU и один skip"
  ],
  [
    "All model {kind}, channel {channel}, {connections} connections from {parameters} unique parameters, shared E in a, Rotate90 b and tied readout roles, {width} channels, one Dense matrix, pre-RMSNorm, ReLU and residual skip, fixed color scale{highlight}",
    "Modelo completo: {kind}, canal {channel}, {connections} conexiones de {parameters} parámetros únicos, E compartida entre a, Rotate90 b y salida, {width} canales, una matriz densa, pre-RMSNorm, ReLU y residual, escala de color fija{highlight}",
    "Gesamtes Modell: {kind}, Kanal {channel}, {connections} Verbindungen aus {parameters} eindeutigen Parametern, gemeinsames E für a, Rotate90 b und Ausgabe, {width} Kanäle, eine dichte Matrix, Pre-RMSNorm, ReLU und Residualpfad, feste Farbskala{highlight}",
    "モデル全体の{kind}、チャネル{channel}、{parameters}個の共有パラメータから{connections}結合。Eをa、Rotate90 b、出力で共有。{width}チャネル、全結合行列1つ、事前RMSNorm、ReLUと残差、固定した色目盛{highlight}",
    "Modèle complet : {kind}, canal {channel}, {connections} connexions pour {parameters} paramètres uniques, E partagé entre a, Rotate90 b et la sortie, {width} canaux, une matrice dense, pre-RMSNorm, ReLU et résidu, échelle de couleur fixe{highlight}",
    "Вся модель: {kind}, канал {channel}, {connections} связей из {parameters} уникальных параметров. E общая для a, Rotate90 b и readout; {width} каналов, одна плотная матрица, pre-RMSNorm, ReLU и skip, фиксированная шкала цвета{highlight}"
  ],
  [
    "Dense + ReLU",
    "Densa + ReLU",
    "Dicht + ReLU",
    "全結合＋ReLU",
    "Dense + ReLU",
    "Плотная + ReLU"
  ],
  [
    "Dense matrix A",
    "Matriz densa A",
    "Dichte Matrix A",
    "全結合行列 A",
    "Matrice dense A",
    "Плотная матрица A"
  ],
  [
    "Input channel ↓ · output channel →",
    "Canal de entrada ↓ · salida →",
    "Eingangskanal ↓ · Ausgangskanal →",
    "入力チャネル ↓ · 出力チャネル →",
    "Canal d’entrée ↓ · sortie →",
    "Входной канал ↓ · выходной канал →"
  ],
  [
    "Dense → ReLU",
    "Densa → ReLU",
    "Dicht → ReLU",
    "全結合 → ReLU",
    "Dense → ReLU",
    "Плотная → ReLU"
  ],
  [
    "Dense → ReLU + skip → RMSNorm",
    "Densa → ReLU + residual → RMSNorm",
    "Dicht → ReLU + Residuum → RMSNorm",
    "全結合 → ReLU＋残差 → RMSNorm",
    "Dense → ReLU + résidu → RMSNorm",
    "Плотная → ReLU + skip → RMSNorm"
  ],
  [
    "{width} → 1 · {width} connections · Dense → ReLU + skip",
    "{width} → 1 · {width} conexiones · Densa → ReLU + residual",
    "{width} → 1 · {width} Verbindungen · Dicht → ReLU + Residuum",
    "{width} → 1 · {width}結合 · 全結合 → ReLU＋残差",
    "{width} → 1 · {width} connexions · Dense → ReLU + résidu",
    "{width} → 1 · {width} связей · Плотная → ReLU + skip"
  ],
  [
    "Compare one shared-embedding Rotate90 Dense architecture at widths 128 and 1024. Train locally in your browser with live loss, accuracy and weights.",
    "Compara una arquitectura Rotate90 densa con embedding compartido y anchuras 128 y 1024. Entrena en tu navegador con pérdida, precisión y pesos en tiempo real.",
    "Vergleiche eine dichte Rotate90-Architektur mit gemeinsamem Embedding in den Breiten 128 und 1024. Trainiere lokal im Browser mit Verlust, Genauigkeit und Gewichten in Echtzeit.",
    "共有埋め込みを使う全結合Rotate90を幅128と1024で比較。ブラウザー内で学習し、損失・正解率・重みをリアルタイムに確認できます。",
    "Comparez une architecture Rotate90 dense à embedding partagé, de largeurs 128 et 1024. Entraînez-la dans votre navigateur avec pertes, exactitude et poids en direct.",
    "Сравните плотную архитектуру Rotate90 с общим эмбеддингом на 128 и 1024 канала. Обучение в браузере с loss, точностью и весами в реальном времени."
  ],
  [
    "One shared E embeds a and b and reads out the answer. Input: E[a] + Rotate90(E[b]). One pre-norm residual branch applies RMSNorm → Dense A → ReLU, then adds the original input. Final RMSNorm and Eᵀ produce 97 logits.",
    "Una E compartida representa a y b y produce la salida. Entrada: E[a] + Rotate90(E[b]). Una rama residual pre-norm aplica RMSNorm → densa A → ReLU y suma la entrada original. RMSNorm final y Eᵀ producen 97 logits.",
    "Ein gemeinsames E bettet a und b ein und liest die Antwort aus. Eingabe: E[a] + Rotate90(E[b]). Ein Pre-Norm-Residualzweig wendet RMSNorm → dichte A → ReLU an und addiert die ursprüngliche Eingabe. Abschließendes RMSNorm und Eᵀ liefern 97 Logits.",
    "共有Eでaとbを埋め込み、答えも出力します。入力は E[a] + Rotate90(E[b])。事前正規化の残差分岐でRMSNorm → 全結合A → ReLUを適用し、元の入力を足します。最終RMSNormとEᵀから97個のロジットを出します。",
    "Un E partagé représente a et b et lit la réponse. Entrée : E[a] + Rotate90(E[b]). Une branche résiduelle pre-norm applique RMSNorm → dense A → ReLU et ajoute l’entrée initiale. RMSNorm final et Eᵀ produisent 97 logits.",
    "Общая E задаёт эмбеддинги a и b и читает ответ. Вход: E[a] + Rotate90(E[b]). Одна residual-ветка с pre-norm применяет RMSNorm → плотную A → ReLU и добавляет исходный вход. Финальная RMSNorm и Eᵀ дают 97 логитов."
  ],
  [
    "One shared E (97 × {width}) embeds a and b and reads out the answer. Input: E[a] + Rotate90(E[b]). One pre-norm residual block applies RMSNorm → Dense A → ReLU, then adds the original input. Final RMSNorm and Eᵀ give 97 logits. E starts normal with variance 1/{width}; A is Kaiming normal with variance 2/{width}; there are no biases or learned norm gains.",
    "Una E compartida (97 × {width}) representa a y b y produce la salida. Entrada: E[a] + Rotate90(E[b]). Un bloque residual pre-norm aplica RMSNorm → densa A → ReLU y suma la entrada original. RMSNorm final y Eᵀ dan 97 logits. E empieza normal con varianza 1/{width}; A usa Kaiming normal con varianza 2/{width}; no hay sesgos ni escalas de normalización aprendidas.",
    "Ein gemeinsames E (97 × {width}) bettet a und b ein und liest die Antwort aus. Eingabe: E[a] + Rotate90(E[b]). Ein Pre-Norm-Residualblock wendet RMSNorm → dichte A → ReLU an und addiert die ursprüngliche Eingabe. Abschließendes RMSNorm und Eᵀ liefern 97 Logits. E startet normal mit Varianz 1/{width}; A ist Kaiming-normal mit Varianz 2/{width}; ohne Bias oder gelernte Normskalierung.",
    "共有E (97 × {width}) がaとbを埋め込み、答えも出力します。入力は E[a] + Rotate90(E[b])。事前正規化の残差ブロックでRMSNorm → 全結合A → ReLUを適用し、元の入力を足します。最終RMSNormとEᵀから97個のロジットを出します。Eは分散1/{width}の正規初期化です。Aは分散2/{width}のKaiming正規初期化で、バイアスや学習する正規化係数はありません。",
    "Un E partagé (97 × {width}) représente a et b et lit la réponse. Entrée : E[a] + Rotate90(E[b]). Un bloc résiduel pre-norm applique RMSNorm → dense A → ReLU et ajoute l’entrée initiale. RMSNorm final et Eᵀ donnent 97 logits. E est initialisé selon une loi normale de variance 1/{width}. A suit une initialisation Kaiming normale de variance 2/{width}, sans biais ni gain de normalisation appris.",
    "Общая E (97 × {width}) задаёт эмбеддинги a и b и читает ответ. Вход: E[a] + Rotate90(E[b]). Один residual-блок с pre-norm применяет RMSNorm → плотную A → ReLU и добавляет исходный вход. Финальная RMSNorm и Eᵀ дают 97 логитов. E инициализируется нормально с дисперсией 1/{width}; A инициализируется Kaiming normal с дисперсией 2/{width}; нет bias и обучаемых коэффициентов нормализации."
  ],
  [
    "Seed 11, Rust/CUDA: train ≥99% at epoch 315; held-out ≥90% at 510 and ≥99% at 605, without a memorization plateau. Only one seed verified; browser crossings may differ.",
    "Semilla 11, Rust/CUDA: entrenamiento ≥99% en la época 315; prueba ≥90% en 510 y ≥99% en 605, sin meseta de memorización. Solo una semilla verificada; los umbrales pueden variar en el navegador.",
    "Seed 11, Rust/CUDA: Training ≥99% in Epoche 315; Test ≥90% bei 510 und ≥99% bei 605, ohne Auswendiglernplateau. Nur ein Seed geprüft; Browser-Schwellen können abweichen.",
    "シード11、Rust/CUDAでは学習99%以上が315エポック、テスト90%以上が510、99%以上が605。暗記の停滞期はありません。検証済みシードは1つで、ブラウザーでは到達時期が異なる場合があります。",
    "Graine 11, Rust/CUDA : entraînement ≥99% à l’époque 315 ; test ≥90% à 510 et ≥99% à 605, sans plateau de mémorisation. Une seule graine vérifiée ; les seuils peuvent différer dans le navigateur.",
    "Seed 11, Rust/CUDA: train ≥99% на эпохе 315; тест ≥90% на 510 и ≥99% на 605, без плато запоминания. Проверен один seed; в браузере моменты пересечения порогов могут отличаться."
  ],
  [
    "Seed 11, Rust/CUDA: train ≥99% and held-out ≤10% at epochs 18–110 (92 epochs); held-out ≥90% at 226 and ≥99% at 270. The 116-epoch rise meets the grokking criterion. Beige: train ≥99%; violet: perfect train; held-out ≤10% in both. Only one seed verified; browser crossings may differ.",
    "Semilla 11, Rust/CUDA: entrenamiento ≥99% y prueba ≤10% en las épocas 18–110 (92 épocas); prueba ≥90% en 226 y ≥99% en 270. El aumento de 116 épocas cumple el criterio de grokking. Beige: entrenamiento ≥99%; violeta: perfecto; prueba ≤10% en ambos. Solo una semilla verificada; los umbrales del navegador pueden variar.",
    "Seed 11, Rust/CUDA: Training ≥99% und Test ≤10% in Epochen 18–110 (92 Epochen); Test ≥90% bei 226 und ≥99% bei 270. Der Anstieg in 116 Epochen erfüllt das Grokking-Kriterium. Beige: Training ≥99%; violett: perfektes Training; beide mit Test ≤10%. Nur ein Seed geprüft; Browser-Schwellen können abweichen.",
    "シード11、Rust/CUDAでは18～110エポック（92エポック）で学習99%以上・テスト10%以下。テスト90%以上が226、99%以上が270。116エポックの上昇でGrokking基準を満たします。ベージュは学習99%以上、紫は完全正解で、どちらもテスト10%以下。検証済みシードは1つで、ブラウザーでは到達時期が異なる場合があります。",
    "Graine 11, Rust/CUDA : entraînement ≥99% et test ≤10% aux époques 18–110 (92 époques) ; test ≥90% à 226 et ≥99% à 270. La montée en 116 époques satisfait le critère de grokking. Beige : entraînement ≥99% ; violet : parfait ; test ≤10% dans les deux. Une seule graine vérifiée ; les seuils du navigateur peuvent différer.",
    "Seed 11, Rust/CUDA: train ≥99% и тест ≤10% на эпохах 18–110 (92 эпохи); тест ≥90% на 226 и ≥99% на 270. Рост за 116 эпоху соответствует критерию grokking. Бежевый: train ≥99%; фиолетовый: train 100%; тест ≤10% в обоих случаях. Проверен один seed; в браузере моменты пересечения порогов могут отличаться."
  ],
  [
    "Seed 11, Rust/CUDA: held-out ≥99% at epoch 605 for 128 channels and 270 for 1024. The wider model has a 92-epoch memorization plateau and a 116-epoch rise to ≥90%. Only one seed verified; browser crossings may differ.",
    "Semilla 11, Rust/CUDA: prueba ≥99% en la época 605 para 128 canales y 270 para 1024. El modelo ancho tiene una meseta de memorización de 92 épocas y un aumento a ≥90% en 116 épocas. Solo una semilla verificada; los umbrales del navegador pueden variar.",
    "Seed 11, Rust/CUDA: Test ≥99% in Epoche 605 bei 128 Kanälen und 270 bei 1024. Das breitere Modell hat ein 92-Epochen-Plateau und steigt in 116 Epochen auf ≥90%. Nur ein Seed geprüft; Browser-Schwellen können abweichen.",
    "シード11、Rust/CUDAでは128チャネルが605エポック、1024が270でテスト99%以上。広いモデルは92エポックの暗記停滞期後、116エポックで90%以上になります。検証済みシードは1つで、ブラウザーでは到達時期が異なる場合があります。",
    "Graine 11, Rust/CUDA : test ≥99% à l’époque 605 pour 128 canaux et 270 pour 1024. Le modèle large a un plateau de mémorisation de 92 époques et monte à ≥90% en 116 époques. Une seule graine vérifiée ; les seuils du navigateur peuvent différer.",
    "Seed 11, Rust/CUDA: тест ≥99% на эпохе 605 для 128 каналов и 270 для 1024. У широкой модели плато запоминания 92 эпохи и рост до ≥90% за 116 эпоху. Проверен один seed; в браузере моменты пересечения порогов могут отличаться."
  ],
  [
    "Only width changes: 128 channels / 28,800 parameters, or 1024 / 1,147,904. Switching model tabs pauses the previous model and preserves its weights, optimizer and curves. Training starts from random initialization on your device.",
    "Solo cambia la anchura: 128 canales / 28.800 parámetros, o 1024 / 1.147.904. Al cambiar de modelo se pausa el anterior y se conservan sus pesos, optimizador y curvas. El entrenamiento empieza desde una inicialización aleatoria en tu dispositivo.",
    "Nur die Breite ändert sich: 128 Kanäle / 28.800 Parameter oder 1024 / 1.147.904. Ein Modellwechsel pausiert das vorherige Modell und behält Gewichte, Optimierer und Kurven. Das Training startet zufällig initialisiert auf deinem Gerät.",
    "変わるのは幅だけです。128チャネル・28,800パラメータ、または1024チャネル・1,147,904パラメータ。タブを切り替えると前のモデルを停止し、重み・最適化状態・曲線を保持します。端末上でランダム初期値から学習します。",
    "Seule la largeur change : 128 canaux / 28 800 paramètres, ou 1024 / 1 147 904. Changer de modèle met le précédent en pause et conserve ses poids, son optimiseur et ses courbes. L’entraînement part d’une initialisation aléatoire sur votre appareil.",
    "Меняется только ширина: 128 каналов / 28 800 параметров или 1024 / 1 147 904. При смене вкладки предыдущая модель ставится на паузу, её веса, оптимизатор и кривые сохраняются. Обучение начинается со случайной инициализации на вашем устройстве."
  ],
  [
    "All draws every E connection in its three shared roles and every A connection. Pink is positive; blue is negative. Color shade follows magnitude on a fixed logarithmic scale. Weights shows values; Change ×4 shows changes from initialization with a fixed scale. Highlight emphasizes a channel. Zoom, drag and Fit explore the graph; Channel shows all W inputs to one ReLU output and its skip. All uses neutral nodes. Channel shows probe activations.",
    "Todo dibuja cada conexión de E en sus tres funciones compartidas y cada conexión de A. Rosa: positivo; azul: negativo. El tono sigue la magnitud en una escala logarítmica fija. Pesos muestra valores; Cambio ×4 muestra diferencias respecto al inicio con escala fija. Resaltar destaca un canal. Zoom, arrastre y Ajustar exploran la red; Canal muestra las W entradas a una salida ReLU y su residual. Todo usa nodos neutros. Canal muestra las activaciones de la entrada seleccionada.",
    "Alle zeigt jede E-Verbindung in drei gemeinsamen Rollen und jede A-Verbindung. Rosa ist positiv; blau negativ. Der Farbton folgt dem Betrag auf einer festen logarithmischen Skala. Gewichte zeigt Werte; Änderung ×4 zeigt Änderungen seit der Initialisierung mit fester Skala. Hervorheben betont einen Kanal. Zoom, Ziehen und Einpassen erkunden das Netz; Kanal zeigt alle W Eingänge zu einem ReLU-Ausgang und seinem Skip. Die Gesamtansicht verwendet neutrale Knoten. Die Kanalansicht zeigt Probeaktivierungen.",
    "全体表示ではEの3つの共有用途とAの全結合を描きます。ピンクは正、青は負。色の濃さは大きさに応じ、対数スケールは固定です。重みは現在値、変化×4は初期値からの差を固定スケールで表示します。強調でチャネルを選択。ズーム・ドラッグ・全体表示で探索し、チャネル表示では1つのReLU出力への全W入力と残差を示します。全体表示のノードは中立色です。チャネル表示では試験入力の活性値を示します。",
    "Tout trace chaque connexion de E dans ses trois rôles partagés et chaque connexion de A. Rose : positif ; bleu : négatif. La nuance suit la valeur absolue sur une échelle logarithmique fixe. Poids affiche les valeurs ; Variation ×4 affiche les écarts depuis l’initialisation à échelle fixe. Surligner accentue un canal. Zoom, glisser et Ajuster explorent le graphe ; Canal montre les W entrées vers une sortie ReLU et son résidu. Tout utilise des nœuds neutres. Canal affiche les activations de l’entrée choisie.",
    "Обзор рисует каждую связь E в трёх общих ролях и каждую связь A. Розовый — положительный вес, синий — отрицательный. Оттенок отражает величину веса по фиксированной логарифмической шкале. Веса показывают значения; Изменение ×4 — разницу с инициализацией при фиксированной шкале. Выделение подчёркивает канал. Масштаб, перетаскивание и Вписать позволяют изучать граф; Канал показывает все W входов одного выхода ReLU и skip. В Обзоре узлы нейтральные. В режиме Канал показаны активации проверочной пары."
  ],
  [
    "Language",
    "Idioma",
    "Sprache",
    "言語",
    "Langue",
    "Язык"
  ],
  [
    "Automatic",
    "Automático",
    "Automatisch",
    "自動",
    "Automatique",
    "Автоматически"
  ],
  [
    "Auto",
    "Auto",
    "Auto",
    "自動",
    "Auto",
    "Авто"
  ],
  [
    "Training",
    "Entrenamiento",
    "Training",
    "学習",
    "Entraînement",
    "Обучение"
  ],
  [
    "Weights",
    "Pesos",
    "Gewichte",
    "重み",
    "Poids",
    "Веса"
  ],
  [
    "Network",
    "Red",
    "Netz",
    "ネットワーク",
    "Réseau",
    "Сеть"
  ],
  [
    "Learning curves",
    "Curvas de aprendizaje",
    "Lernkurven",
    "学習曲線",
    "Courbes d’apprentissage",
    "Кривые обучения"
  ],
  [
    "Loss",
    "Pérdida",
    "Verlust",
    "損失",
    "Perte",
    "Ошибка"
  ],
  [
    "Accuracy",
    "Precisión",
    "Genauigkeit",
    "正解率",
    "Exactitude",
    "Точность"
  ],
  [
    "Held-out",
    "Prueba",
    "Test",
    "テスト",
    "Test",
    "Тест"
  ],
  [
    "Cross-entropy · logarithmic scale",
    "Entropía cruzada · escala logarítmica",
    "Kreuzentropie · logarithmische Skala",
    "交差エントロピー・対数目盛",
    "Entropie croisée · échelle logarithmique",
    "Кросс-энтропия · логарифмическая шкала"
  ],
  [
    "Complete-split accuracy",
    "Precisión de la partición completa",
    "Genauigkeit auf dem gesamten Datensatz",
    "全データでの正解率",
    "Exactitude sur toutes les données",
    "Точность на всей выборке"
  ],
  [
    "Evaluated every epoch",
    "Evaluación en cada época",
    "Auswertung nach jeder Epoche",
    "各エポックで評価",
    "Évaluation à chaque époque",
    "Оценка каждую эпоху"
  ],
  [
    "Evaluated every 25 epochs",
    "Evaluación cada 25 épocas",
    "Auswertung alle 25 Epochen",
    "25エポックごとに評価",
    "Évaluation toutes les 25 époques",
    "Оценка каждые 25 эпох"
  ],
  [
    "{count} observations",
    "{count} observaciones",
    "{count} Messungen",
    "評価 {count} 回",
    "{count} mesures",
    "Измерений: {count}"
  ],
  [
    "0 observations",
    "0 observaciones",
    "0 Messungen",
    "評価 0 回",
    "0 mesures",
    "Измерений: 0"
  ],
  [
    "EPOCH",
    "ÉPOCA",
    "EPOCHE",
    "エポック",
    "ÉPOQUE",
    "ЭПОХА"
  ],
  [
    "Epoch",
    "Época",
    "Epoche",
    "エポック",
    "Époque",
    "Эпоха"
  ],
  [
    "Start training to see the curve.",
    "Inicia el entrenamiento para ver la curva.",
    "Starte das Training, um die Kurve zu sehen.",
    "学習を開始すると曲線が表示されます。",
    "Lancez l’entraînement pour voir la courbe.",
    "Запустите обучение, чтобы увидеть кривую."
  ],
  [
    "Answers becoming correct",
    "Respuestas que se vuelven correctas",
    "Antworten werden richtig",
    "正解に変わる過程",
    "Les réponses deviennent correctes",
    "Как ответы становятся верными"
  ],
  [
    "Wrong",
    "Incorrecto",
    "Falsch",
    "不正解",
    "Incorrect",
    "Неверно"
  ],
  [
    "Correct",
    "Correcto",
    "Richtig",
    "正解",
    "Correct",
    "Верно"
  ],
  [
    "correct",
    "correcto",
    "richtig",
    "正解",
    "correct",
    "верно"
  ],
  [
    "not yet correct",
    "todavía incorrecto",
    "noch falsch",
    "まだ不正解",
    "encore incorrect",
    "пока неверно"
  ],
  [
    "128 held-out equations",
    "128 ecuaciones de prueba",
    "128 Testgleichungen",
    "テスト用の128式",
    "128 équations de test",
    "128 тестовых уравнений"
  ],
  [
    "128 fixed held-out equations",
    "128 ecuaciones de prueba fijas",
    "128 feste Testgleichungen",
    "固定した128個のテスト式",
    "128 équations de test fixes",
    "128 фиксированных тестовых примеров"
  ],
  [
    "Sample: {correct}/{count} correct",
    "Muestra: {correct}/{count} correctas",
    "Stichprobe: {correct}/{count} richtig",
    "標本：{correct}/{count} 正解",
    "Échantillon : {correct}/{count} correctes",
    "Примеры: {correct}/{count} верно"
  ],
  [
    "{correct}/{count} correct",
    "{correct}/{count} correctas",
    "{correct}/{count} richtig",
    "{correct}/{count} 正解",
    "{correct}/{count} correctes",
    "{correct}/{count} верно"
  ],
  [
    "Snapshot epoch {epoch}",
    "Captura de la época {epoch}",
    "Aufnahme: Epoche {epoch}",
    "スナップショット：{epoch} エポック",
    "Capture à l’époque {epoch}",
    "Снимок на эпохе {epoch}"
  ],
  [
    "Snapshot epoch 0",
    "Captura de la época 0",
    "Aufnahme: Epoche 0",
    "スナップショット：0 エポック",
    "Capture à l’époque 0",
    "Снимок на эпохе 0"
  ],
  [
    "Preparing probes",
    "Preparando ejemplos",
    "Testbeispiele werden vorbereitet",
    "テスト式を準備中",
    "Préparation des exemples",
    "Подготовка примеров"
  ],
  [
    "Preparing held-out probes…",
    "Preparando ejemplos de prueba…",
    "Testbeispiele werden vorbereitet…",
    "テスト式を準備中…",
    "Préparation des exemples de test…",
    "Подготовка тестовых примеров…"
  ],
  [
    "Captured epochs →",
    "Épocas capturadas →",
    "Aufgenommene Epochen →",
    "記録したエポック →",
    "Époques enregistrées →",
    "Эпохи снимков →"
  ],
  [
    "Click a row to inspect an equation. Complete-split accuracy is shown in Training desk.",
    "Haz clic en una fila para ver la ecuación. La precisión total aparece en el panel de entrenamiento.",
    "Klicke auf eine Zeile, um die Gleichung zu prüfen. Die Gesamtgenauigkeit steht im Trainingsbereich.",
    "行をクリックすると式を確認できます。全テストの正解率は学習パネルに表示されます。",
    "Cliquez sur une ligne pour examiner l’équation. L’exactitude totale apparaît dans le panneau d’entraînement.",
    "Нажмите на строку, чтобы изучить пример. Точность на всей выборке указана в панели обучения."
  ],
  [
    "All weights",
    "Todos los pesos",
    "Alle Gewichte",
    "すべての重み",
    "Tous les poids",
    "Все веса"
  ],
  [
    "All weight changes",
    "Todos los cambios de peso",
    "Alle Gewichtsänderungen",
    "すべての重みの変化",
    "Toutes les variations de poids",
    "Все изменения весов"
  ],
  [
    "All",
    "Todos",
    "Alle",
    "全体",
    "Tous",
    "Все"
  ],
  [
    "Channel",
    "Canal",
    "Kanal",
    "チャネル",
    "Canal",
    "Канал"
  ],
  [
    "Change",
    "Cambio",
    "Änderung",
    "変化",
    "Variation",
    "Изменение"
  ],
  [
    "Change ×4",
    "Cambio ×4",
    "Änderung ×4",
    "変化 ×4",
    "Variation ×4",
    "Изменение ×4"
  ],
  [
    "Highlight",
    "Resaltar",
    "Hervorheben",
    "強調",
    "Surligner",
    "Выделить"
  ],
  [
    "Fit",
    "Ajustar",
    "Einpassen",
    "全体表示",
    "Ajuster",
    "Вместить"
  ],
  [
    "Weight atlas",
    "Atlas de pesos",
    "Gewichtsatlas",
    "重みマップ",
    "Atlas des poids",
    "Атлас весов"
  ],
  [
    "Input / output",
    "Entrada / salida",
    "Eingabe / Ausgabe",
    "入力・出力",
    "Entrée / sortie",
    "Вход / выход"
  ],
  [
    "Center R",
    "Centrar R",
    "R zentrieren",
    "Rを中心化",
    "Centrer R",
    "Центрировать R"
  ],
  [
    "View",
    "Vista",
    "Ansicht",
    "表示",
    "Vue",
    "Вид"
  ],
  [
    "Shared embedding E",
    "Embedding compartido E",
    "Gemeinsames Embedding E",
    "共有埋め込み E",
    "Embedding partagé E",
    "Общий эмбеддинг E"
  ],
  [
    "Shared embedding",
    "Embedding compartido",
    "Gemeinsames Embedding",
    "共有埋め込み",
    "Embedding partagé",
    "Общий эмбеддинг"
  ],
  [
    "Rotate90(E) · b role",
    "Rotate90(E) · papel de b",
    "Rotate90(E) · Rolle von b",
    "Rotate90(E)・bの役割",
    "Rotate90(E) · rôle de b",
    "Rotate90(E) · роль b"
  ],
  [
    "Tied readout Eᵀ",
    "Salida compartida Eᵀ",
    "Gebundene Ausgabe Eᵀ",
    "共有出力 Eᵀ",
    "Sortie partagée Eᵀ",
    "Выходные веса Eᵀ"
  ],
  [
    "Tied readout",
    "Salida compartida",
    "Gebundene Ausgabe",
    "共有出力",
    "Sortie partagée",
    "Выходные веса"
  ],
  [
    "Residue ↓ · channel →",
    "Residuo ↓ · canal →",
    "Rest ↓ · Kanal →",
    "剰余 ↓・チャネル →",
    "Reste ↓ · canal →",
    "Остаток ↓ · канал →"
  ],
  [
    "Answer ↓ · channel →",
    "Respuesta ↓ · canal →",
    "Antwort ↓ · Kanal →",
    "答え ↓・チャネル →",
    "Réponse ↓ · canal →",
    "Ответ ↓ · канал →"
  ],
  [
    "Layer ↓ · output channel →",
    "Capa ↓ · canal de salida →",
    "Schicht ↓ · Ausgabekanal →",
    "層 ↓・出力チャネル →",
    "Couche ↓ · canal de sortie →",
    "Слой ↓ · выходной канал →"
  ],
  [
    "Channel scales",
    "Escalas de canales",
    "Kanalskalen",
    "チャネルの尺度",
    "Échelles des canaux",
    "Масштабы каналов"
  ],
  [
    "Probe contributions",
    "Contribuciones al ejemplo",
    "Beiträge zum Testbeispiel",
    "テスト式への寄与",
    "Contributions à l’exemple",
    "Вклады в ответ"
  ],
  [
    "Top 16: —",
    "16 mayores: —",
    "Top 16: —",
    "上位16：—",
    "16 premiers : —",
    "Топ-16: —"
  ],
  [
    "Top 16: {percent}%",
    "16 mayores: {percent}%",
    "Top 16: {percent}%",
    "上位16：{percent}%",
    "16 premiers : {percent}%",
    "Топ-16: {percent}%"
  ],
  [
    "Probe the model",
    "Probar el modelo",
    "Modell prüfen",
    "モデルを試す",
    "Tester le modèle",
    "Проверить модель"
  ],
  [
    "Probe this network's current prediction.",
    "Consulta la predicción actual de la red.",
    "Prüfe die aktuelle Vorhersage des Netzes.",
    "現在のネットワークの予測を確認します。",
    "Examinez la prédiction actuelle du réseau.",
    "Проверьте текущий ответ сети."
  ],
  [
    "Use integer a = 0…96 and b = 1…96.",
    "Usa enteros a = 0…96 y b = 1…96.",
    "Verwende ganze Zahlen a = 0…96 und b = 1…96.",
    "整数 a = 0…96、b = 1…96 を入力してください。",
    "Utilisez des entiers a = 0…96 et b = 1…96.",
    "Введите целые a = 0…96 и b = 1…96."
  ],
  [
    "Correct answer: {answer} · confidence {percent}% · {result}",
    "Respuesta correcta: {answer} · confianza {percent}% · {result}",
    "Richtige Antwort: {answer} · Konfidenz {percent}% · {result}",
    "正しい答え：{answer}・確信度 {percent}%・{result}",
    "Bonne réponse : {answer} · confiance {percent}% · {result}",
    "Правильный ответ: {answer} · уверенность {percent}% · {result}"
  ],
  [
    "Training desk",
    "Panel de entrenamiento",
    "Trainingsbereich",
    "学習パネル",
    "Panneau d’entraînement",
    "Панель обучения"
  ],
  [
    "Train accuracy",
    "Precisión de entrenamiento",
    "Trainingsgenauigkeit",
    "学習の正解率",
    "Exactitude d’entraînement",
    "Точность обучения"
  ],
  [
    "Held-out accuracy",
    "Precisión de prueba",
    "Testgenauigkeit",
    "テストの正解率",
    "Exactitude de test",
    "Точность теста"
  ],
  [
    "Train loss",
    "Pérdida de entrenamiento",
    "Trainingsverlust",
    "学習の損失",
    "Perte d’entraînement",
    "Ошибка обучения"
  ],
  [
    "Held-out loss",
    "Pérdida de prueba",
    "Testverlust",
    "テストの損失",
    "Perte de test",
    "Ошибка теста"
  ],
  [
    "Seed",
    "Semilla",
    "Seed",
    "乱数シード",
    "Graine",
    "Seed"
  ],
  [
    "Epoch limit",
    "Límite de épocas",
    "Epochenlimit",
    "エポック上限",
    "Limite d’époques",
    "Лимит эпох"
  ],
  [
    "Training engine",
    "Motor de entrenamiento",
    "Trainings-Engine",
    "学習エンジン",
    "Moteur d’entraînement",
    "Движок обучения"
  ],
  [
    "Stop at 99% held-out accuracy",
    "Detener al 99% de precisión de prueba",
    "Bei 99% Testgenauigkeit stoppen",
    "テスト正解率99%で停止",
    "Arrêter à 99% d’exactitude de test",
    "Остановить при 99% на тесте"
  ],
  [
    "Start training",
    "Iniciar entrenamiento",
    "Training starten",
    "学習を開始",
    "Lancer l’entraînement",
    "Начать обучение"
  ],
  [
    "Pause training",
    "Pausar entrenamiento",
    "Training pausieren",
    "学習を一時停止",
    "Mettre en pause",
    "Пауза обучения"
  ],
  [
    "+1 epoch",
    "+1 época",
    "+1 Epoche",
    "+1 エポック",
    "+1 époque",
    "+1 эпоха"
  ],
  [
    "+25 epochs",
    "+25 épocas",
    "+25 Epochen",
    "+25 エポック",
    "+25 époques",
    "+25 эпох"
  ],
  [
    "Reset",
    "Reiniciar",
    "Zurücksetzen",
    "リセット",
    "Réinitialiser",
    "Сбросить"
  ],
  [
    "Weight updates",
    "Actualización de pesos",
    "Gewichtsaufnahmen",
    "重みの更新表示",
    "Actualisation des poids",
    "Снимки весов"
  ],
  [
    "1 / second",
    "1 / segundo",
    "1 / Sekunde",
    "1 / 秒",
    "1 / seconde",
    "1 / секунду"
  ],
  [
    "5 / second",
    "5 / segundo",
    "5 / Sekunde",
    "5 / 秒",
    "5 / seconde",
    "5 / секунду"
  ],
  [
    "10 / second",
    "10 / segundo",
    "10 / Sekunde",
    "10 / 秒",
    "10 / seconde",
    "10 / секунду"
  ],
  [
    "Ready when you are",
    "Listo para comenzar",
    "Bereit zum Start",
    "開始できます",
    "Prêt à commencer",
    "Готово к запуску"
  ],
  [
    "{count} updates / sec",
    "{count} actualizaciones / s",
    "{count} Updates / s",
    "{count} 更新 / 秒",
    "{count} mises à jour / s",
    "{count} обновлений / с"
  ],
  [
    "Preparing training engine",
    "Preparando el motor",
    "Trainings-Engine wird vorbereitet",
    "学習エンジンを準備中",
    "Préparation du moteur",
    "Подготовка движка"
  ],
  [
    "Preparing training engine…",
    "Preparando el motor…",
    "Trainings-Engine wird vorbereitet…",
    "学習エンジンを準備中…",
    "Préparation du moteur…",
    "Подготовка движка…"
  ],
  [
    "Training engine could not start",
    "No se pudo iniciar el motor",
    "Trainings-Engine konnte nicht starten",
    "学習エンジンを起動できませんでした",
    "Le moteur n’a pas pu démarrer",
    "Движок не удалось запустить"
  ],
  [
    "Waiting for the first observation",
    "Esperando la primera evaluación",
    "Warten auf die erste Messung",
    "最初の評価を待機中",
    "En attente de la première mesure",
    "Ожидание первого измерения"
  ],
  [
    "Export metrics",
    "Exportar métricas",
    "Messwerte exportieren",
    "指標を出力",
    "Exporter les mesures",
    "Экспорт метрик"
  ],
  [
    "Save weights",
    "Guardar pesos",
    "Gewichte speichern",
    "重みを保存",
    "Enregistrer les poids",
    "Сохранить веса"
  ],
  [
    "Local training · data stays on your device",
    "Entrenamiento local · los datos quedan en tu dispositivo",
    "Lokales Training · Daten bleiben auf deinem Gerät",
    "ローカル学習・データは端末内に保存",
    "Entraînement local · données sur votre appareil",
    "Локальное обучение · данные остаются на устройстве"
  ],
  [
    "Loading",
    "Cargando",
    "Laden",
    "読み込み中",
    "Chargement",
    "Загрузка"
  ],
  [
    "Ready",
    "Listo",
    "Bereit",
    "準備完了",
    "Prêt",
    "Готово"
  ],
  [
    "Paused",
    "En pausa",
    "Pausiert",
    "一時停止",
    "En pause",
    "Пауза"
  ],
  [
    "Pausing",
    "Pausando",
    "Wird pausiert",
    "停止中",
    "Mise en pause",
    "Останавливается"
  ],
  [
    "Unavailable",
    "No disponible",
    "Nicht verfügbar",
    "利用不可",
    "Indisponible",
    "Недоступно"
  ],
  [
    "Error",
    "Error",
    "Fehler",
    "エラー",
    "Erreur",
    "Ошибка"
  ],
  [
    "Target reached",
    "Objetivo alcanzado",
    "Ziel erreicht",
    "目標達成",
    "Objectif atteint",
    "Цель достигнута"
  ],
  [
    "Grokking reached",
    "Grokking alcanzado",
    "Grokking erreicht",
    "Grokking達成",
    "Grokking atteint",
    "Grokking достигнут"
  ],
  [
    "Epoch limit reached",
    "Límite de épocas alcanzado",
    "Epochenlimit erreicht",
    "エポック上限に到達",
    "Limite d’époques atteinte",
    "Достигнут лимит эпох"
  ],
  [
    "Learning",
    "Aprendiendo",
    "Lernen",
    "学習中",
    "Apprentissage",
    "Учится"
  ],
  [
    "Generalizing",
    "Generalizando",
    "Verallgemeinerung",
    "汎化が進行中",
    "Généralisation",
    "Начинает обобщать"
  ],
  [
    "Generalized",
    "Generalizado",
    "Verallgemeinert",
    "汎化を達成",
    "Généralisation atteinte",
    "Обобщение достигнуто"
  ],
  [
    "Memorization plateau observed",
    "Meseta de memorización observada",
    "Plateau nach Auswendiglernen beobachtet",
    "暗記後の停滞期を観測",
    "Plateau de mémorisation observé",
    "Наблюдалось плато запоминания"
  ],
  [
    "Short-plateau grokking",
    "Grokking con meseta corta",
    "Grokking nach kurzem Plateau",
    "短い停滞期のGrokking",
    "Grokking après un court plateau",
    "Grokking после короткого плато"
  ],
  [
    "Delayed grokking",
    "Grokking tardío",
    "Verzögertes Grokking",
    "遅れて生じるGrokking",
    "Grokking tardif",
    "Поздний grokking"
  ],
  [
    "{width} channels · {parameters} parameters · {activation} · FP32",
    "{width} canales · {parameters} parámetros · {activation} · FP32",
    "{width} Kanäle · {parameters} Parameter · {activation} · FP32",
    "{width} チャネル・{parameters} パラメータ・{activation}・FP32",
    "{width} canaux · {parameters} paramètres · {activation} · FP32",
    "{width} каналов · {parameters} параметров · {activation} · FP32"
  ],
  [
    "{connections} connections · {parameters} parameters",
    "{connections} conexiones · {parameters} parámetros",
    "{connections} Verbindungen · {parameters} Parameter",
    "{connections} 結合・{parameters} パラメータ",
    "{connections} connexions · {parameters} paramètres",
    "{connections} связей · {parameters} параметров"
  ],
  [
    "CPU drawing",
    "Dibujo en CPU",
    "CPU-Darstellung",
    "CPU描画",
    "Rendu CPU",
    "Отрисовка на CPU"
  ],
  [
    "fixed scale",
    "escala fija",
    "feste Skala",
    "固定目盛",
    "échelle fixe",
    "фиксированный масштаб"
  ],
  [
    "Contribution: {value}",
    "Contribución: {value}",
    "Beitrag: {value}",
    "寄与：{value}",
    "Contribution : {value}",
    "Вклад: {value}"
  ],
  [
    "Contribution: —",
    "Contribución: —",
    "Beitrag: —",
    "寄与：—",
    "Contribution : —",
    "Вклад: —"
  ],
  [
    "Block {block} · RMS {value}",
    "Bloque {block} · RMS {value}",
    "Block {block} · RMS {value}",
    "ブロック {block}・RMS {value}",
    "Bloc {block} · RMS {value}",
    "Блок {block} · RMS {value}"
  ],
  [
    "Block 1 · RMS —",
    "Bloque 1 · RMS —",
    "Block 1 · RMS —",
    "ブロック1・RMS —",
    "Bloc 1 · RMS —",
    "Блок 1 · RMS —"
  ],
  [
    "probe activations · click a node",
    "activaciones del ejemplo · haz clic en un nodo",
    "Aktivierungen · Knoten anklicken",
    "テスト式の活性値・ノードをクリック",
    "activations de l’exemple · cliquez sur un nœud",
    "активации примера · нажмите на узел"
  ],
  [
    "Tied Eᵀ · 97 answers",
    "Eᵀ compartida · 97 respuestas",
    "Gebundenes Eᵀ · 97 Antworten",
    "共有 Eᵀ・97個の答え",
    "Eᵀ partagé · 97 réponses",
    "Общая Eᵀ · 97 ответов"
  ],
  [
    "Final RMSNorm h",
    "RMSNorm final h",
    "Letztes RMSNorm h",
    "最終RMSNorm h",
    "RMSNorm final h",
    "Финальная RMSNorm h"
  ],
  [
    "skip",
    "residual",
    "Residuum",
    "残差",
    "résidu",
    "skip"
  ],
  [
    "activation",
    "activación",
    "Aktivierung",
    "活性値",
    "activation",
    "активация"
  ],
  [
    "click to select",
    "haz clic para seleccionar",
    "zum Auswählen anklicken",
    "クリックして選択",
    "cliquez pour sélectionner",
    "нажмите для выбора"
  ],
  [
    "weight",
    "peso",
    "Gewicht",
    "重み",
    "poids",
    "вес"
  ],
  [
    "raw weight",
    "peso original",
    "rohes Gewicht",
    "生の重み",
    "poids brut",
    "исходный вес"
  ],
  [
    "rotated E",
    "E rotada",
    "rotiertes E",
    "回転したE",
    "E tourné",
    "повёрнутая E"
  ],
  [
    "answer class",
    "clase de respuesta",
    "Antwortklasse",
    "答えのクラス",
    "classe de réponse",
    "класс ответа"
  ],
  [
    "residue",
    "residuo",
    "Rest",
    "剰余",
    "reste",
    "остаток"
  ],
  [
    "About this experiment",
    "Acerca del experimento",
    "Über dieses Experiment",
    "この実験について",
    "À propos de l’expérience",
    "Об эксперименте"
  ],
  [
    "What counts as grokking?",
    "¿Qué se considera grokking?",
    "Was zählt als Grokking?",
    "Grokkingの判定条件",
    "Quand parle-t-on de grokking ?",
    "Что считается grokking?"
  ],
  [
    "One architecture, two widths",
    "Una arquitectura, dos anchuras",
    "Eine Architektur, zwei Breiten",
    "同じ構造、2種類の幅",
    "Une architecture, deux largeurs",
    "Одна архитектура, две ширины"
  ],
  [
    "All weights, live",
    "Todos los pesos en tiempo real",
    "Alle Gewichte in Echtzeit",
    "すべての重みをリアルタイム表示",
    "Tous les poids en direct",
    "Все веса в реальном времени"
  ],
  [
    "Task and solution",
    "Problema y solución",
    "Aufgabe und Lösung",
    "課題と解き方",
    "Problème et solution",
    "Задача и решение"
  ],
  [
    "Division modulo 97",
    "División módulo 97",
    "Division modulo 97",
    "97を法とする除算",
    "Division modulo 97",
    "Деление по модулю 97"
  ],
  [
    "Task",
    "Problema",
    "Aufgabe",
    "課題",
    "Problème",
    "Задача"
  ],
  [
    "Solution",
    "Solución",
    "Lösung",
    "解き方",
    "Solution",
    "Решение"
  ],
  [
    "Examples",
    "Ejemplos",
    "Beispiele",
    "例",
    "Exemples",
    "Примеры"
  ],
  [
    "Model",
    "Modelo",
    "Modell",
    "モデル",
    "Modèle",
    "Модель"
  ],
  [
    "Optimizer preset",
    "Ajuste del optimizador",
    "Optimierer-Voreinstellung",
    "最適化の設定",
    "Réglage de l’optimiseur",
    "Режим оптимизатора"
  ],
  [
    "Fast",
    "Rápido",
    "Schnell",
    "高速",
    "Rapide",
    "Быстрый"
  ],
  [
    "Slow",
    "Lento",
    "Langsam",
    "低速",
    "Lent",
    "Медленный"
  ],
  [
    "Fast optimizer: reaches 99% held-out accuracy earliest on every Rust/CUDA seed",
    "Optimizador rápido: alcanza el 99% en prueba antes que ningún otro en todas las semillas de Rust/CUDA",
    "Schneller Optimierer: erreicht 99% Testgenauigkeit am frühesten bei jedem Rust/CUDA-Seed",
    "高速な最適化：Rust/CUDAのどのシードでもテスト99%に最も早く到達",
    "Optimiseur rapide : atteint 99% au test le plus tôt sur chaque graine Rust/CUDA",
    "Быстрый оптимизатор: раньше всех достигает 99% на тесте на каждом seed в Rust/CUDA"
  ],
  [
    "Slow optimizer: reaches 99% held-out accuracy latest, yet on every Rust/CUDA seed",
    "Optimizador lento: alcanza el 99% en prueba más tarde, pero en todas las semillas de Rust/CUDA",
    "Langsamer Optimierer: erreicht 99% Testgenauigkeit am spätesten, aber bei jedem Rust/CUDA-Seed",
    "低速な最適化：テスト99%への到達は最も遅いが、Rust/CUDAのどのシードでも到達",
    "Optimiseur lent : atteint 99% au test le plus tard, mais sur chaque graine Rust/CUDA",
    "Медленный оптимизатор: позже всех достигает 99% на тесте, но на каждом seed в Rust/CUDA"
  ],
  [
    "Plateau",
    "Meseta",
    "Plateau",
    "停滞期",
    "Plateau",
    "Плато"
  ],
  [
    "Plateau optimizer: the longest memorization plateau before 99% held-out accuracy on every Rust/CUDA seed; where Slow has it, the longest followed by grokking",
    "Optimizador de meseta: la meseta de memorización más larga antes del 99% en prueba en todas las semillas de Rust/CUDA; si es la de Lento, la más larga seguida de grokking",
    "Plateau-Optimierer: das längste Auswendiglernplateau vor 99% Testgenauigkeit bei jedem Rust/CUDA-Seed; hat Langsam es, das längste mit anschließendem Grokking",
    "停滞期の最適化：Rust/CUDAのどのシードでもテスト99%到達前の暗記停滞期が最も長い。それが低速の場合は、grokkingが続く停滞期のうち最も長い",
    "Optimiseur plateau : le plus long plateau de mémorisation avant 99% au test sur chaque graine Rust/CUDA ; si c’est celui de Lent, le plus long suivi d’un grokking",
    "Оптимизатор с плато: самое долгое плато запоминания перед 99% на тесте на каждом seed в Rust/CUDA; если оно у «Медленного», то самое долгое из тех, за которыми следует grokking"
  ],
  [
    "Memorization plateau: {epochs} epochs.",
    "Meseta de memorización: {epochs} épocas.",
    "Auswendiglernplateau: {epochs} Epochen.",
    "暗記停滞期のエポック数：{epochs}。",
    "Plateau de mémorisation : {epochs} époques.",
    "Длина плато запоминания в эпохах: {epochs}."
  ],
  [
    "Width",
    "Anchura",
    "Breite",
    "幅",
    "Largeur",
    "Ширина"
  ],
  [
    "or",
    "o",
    "oder",
    "または",
    "ou",
    "или"
  ],
  [
    "Rotate90 model architecture",
    "Arquitectura del modelo Rotate90",
    "Architektur des Rotate90-Modells",
    "Rotate90モデルの構造",
    "Architecture du modèle Rotate90",
    "Архитектура модели Rotate90"
  ],
  [
    "Shared E",
    "E compartida",
    "Gemeinsames E",
    "共有E",
    "E partagé",
    "Общая E"
  ],
  [
    "Residual block",
    "Bloque residual",
    "Residualblock",
    "残差ブロック",
    "Bloc résiduel",
    "Остаточный блок"
  ],
  [
    "Skip connection",
    "Conexión residual",
    "Direktverbindung",
    "スキップ接続",
    "Connexion résiduelle",
    "Обходная связь"
  ],
  [
    "97 logits",
    "97 logits",
    "97 Logits",
    "97個のロジット",
    "97 logits",
    "97 логитов"
  ],
  [
    "Dashed lines reuse E.",
    "Las líneas discontinuas reutilizan E.",
    "Gestrichelte Linien verwenden dasselbe E.",
    "破線は同じEの共有を表します。",
    "Les pointillés réutilisent le même E.",
    "Пунктир обозначает повторное использование E."
  ],
  [
    "Model presentation",
    "Vista del modelo",
    "Modelldarstellung",
    "モデルの表示",
    "Présentation du modèle",
    "Представление модели"
  ],
  [
    "Diagram",
    "Diagrama",
    "Diagramm",
    "図",
    "Schéma",
    "Схема"
  ],
  [
    "Types",
    "Tipos",
    "Typen",
    "型",
    "Types",
    "Типы"
  ],
  [
    "Equations",
    "Ecuaciones",
    "Gleichungen",
    "数式",
    "Équations",
    "Формулы"
  ],
  [
    "Symbol",
    "Símbolo",
    "Symbol",
    "記号",
    "Symbole",
    "Символ"
  ],
  [
    "Type / domain",
    "Tipo / dominio",
    "Typ / Wertebereich",
    "型・範囲",
    "Type / domaine",
    "Тип / область"
  ],
  [
    "Forward pass",
    "Propagación hacia delante",
    "Vorwärtsrechnung",
    "順伝播",
    "Propagation avant",
    "Прямой проход"
  ],
  [
    "Fixed operations",
    "Operaciones fijas",
    "Feste Operationen",
    "固定演算",
    "Opérations fixes",
    "Фиксированные операции"
  ],
  [
    "ℝ denotes real values; vectors are rows. q is the correct answer, q̂ the prediction. Training uses FP32.",
    "ℝ representa valores reales; los vectores son filas. q es la respuesta correcta y q̂ la predicción. El entrenamiento usa FP32.",
    "ℝ bezeichnet reelle Werte; Vektoren sind Zeilen. q ist die richtige Antwort, q̂ die Vorhersage. Das Training verwendet FP32.",
    "ℝは実数を表し、ベクトルは行ベクトルです。qは正解、q̂は予測です。学習ではFP32を使います。",
    "ℝ désigne les valeurs réelles ; les vecteurs sont des lignes. q est la bonne réponse, q̂ la prédiction. L’entraînement utilise FP32.",
    "ℝ — вещественные числа; векторы записаны строками. q — правильный ответ, q̂ — предсказание. Обучение использует FP32."
  ],
  [
    "Open lab",
    "Abrir laboratorio",
    "Labor öffnen",
    "実験を開く",
    "Ouvrir le labo",
    "Открыть лабораторию"
  ],
  [
    "Close",
    "Cerrar",
    "Schließen",
    "閉じる",
    "Fermer",
    "Закрыть"
  ],
  [
    "Next",
    "Siguiente",
    "Weiter",
    "次へ",
    "Suivant",
    "Далее"
  ],
  [
    "Back",
    "Atrás",
    "Zurück",
    "戻る",
    "Retour",
    "Назад"
  ],
  [
    "Equation",
    "Ecuación",
    "Gleichung",
    "式",
    "Équation",
    "Уравнение"
  ],
  [
    "Check",
    "Comprobación",
    "Prüfung",
    "検算",
    "Vérification",
    "Проверка"
  ],
  [
    "Inputs",
    "Entradas",
    "Eingaben",
    "入力",
    "Entrées",
    "Вход"
  ],
  [
    "Find the inverse",
    "Encontrar el inverso",
    "Inverse bestimmen",
    "逆元を求める",
    "Trouver l’inverse",
    "Найти обратный элемент"
  ],
  [
    "Multiply and take the remainder",
    "Multiplicar y tomar el residuo",
    "Multiplizieren und Rest nehmen",
    "掛け算して剰余を求める",
    "Multiplier et prendre le reste",
    "Умножить и взять остаток"
  ],
  [
    "Verify the answer",
    "Verificar la respuesta",
    "Antwort prüfen",
    "答えを検算",
    "Vérifier la réponse",
    "Проверить ответ"
  ],
  [
    "Answer",
    "Respuesta",
    "Antwort",
    "答え",
    "Réponse",
    "Ответ"
  ],
  [
    "Mathematical solution",
    "Solución matemática",
    "Mathematische Lösung",
    "数学的な解き方",
    "Solution mathématique",
    "Математическое решение"
  ],
  [
    "How the model learns",
    "Cómo aprende el modelo",
    "Wie das Modell lernt",
    "モデルの学習方法",
    "Comment le modèle apprend",
    "Как учится модель"
  ],
  [
    "9,312 equations",
    "9.312 ecuaciones",
    "9.312 Gleichungen",
    "9,312個の式",
    "9 312 équations",
    "9 312 уравнений"
  ],
  [
    "4,656 train · 4,656 test",
    "4.656 entrenamiento · 4.656 prueba",
    "4.656 Training · 4.656 Test",
    "学習4,656式・テスト4,656式",
    "4 656 entraînement · 4 656 test",
    "4 656 обучение · 4 656 тест"
  ],
  [
    "CPU · WASM scalar",
    "CPU · WASM escalar",
    "CPU · skalares WASM",
    "CPU・スカラーWASM",
    "CPU · WASM scalaire",
    "CPU · скалярный WASM"
  ],
  [
    "CPU · WASM SIMD128",
    "CPU · WASM SIMD128",
    "CPU · WASM SIMD128",
    "CPU・WASM SIMD128",
    "CPU · WASM SIMD128",
    "CPU · WASM SIMD128"
  ],
  [
    "This browser does not support WebAssembly.",
    "Este navegador no admite WebAssembly.",
    "Dieser Browser unterstützt WebAssembly nicht.",
    "このブラウザーはWebAssemblyに対応していません。",
    "Ce navigateur ne prend pas en charge WebAssembly.",
    "Этот браузер не поддерживает WebAssembly."
  ],
  [
    "The WASM trainer is not supported by this browser.",
    "Este navegador no admite el motor WASM.",
    "Dieser Browser unterstützt den WASM-Trainer nicht.",
    "このブラウザーはWASM学習エンジンに対応していません。",
    "Ce navigateur ne prend pas en charge le moteur WASM.",
    "Этот браузер не поддерживает движок WASM."
  ],
  [
    "Could not load WASM trainer: HTTP {status}.",
    "No se pudo cargar el motor WASM: HTTP {status}.",
    "WASM-Trainer konnte nicht geladen werden: HTTP {status}.",
    "WASM学習エンジンを読み込めませんでした：HTTP {status}。",
    "Impossible de charger le moteur WASM : HTTP {status}.",
    "Не удалось загрузить движок WASM: HTTP {status}."
  ],
  [
    "Training diverged: metrics are not finite. Reset the experiment.",
    "El entrenamiento divergió: métricas no finitas. Reinicia el experimento.",
    "Training divergiert: Messwerte sind nicht endlich. Setze das Experiment zurück.",
    "学習が発散しました。指標が有限値ではありません。実験をリセットしてください。",
    "L’entraînement a divergé : mesures non finies. Réinitialisez l’expérience.",
    "Обучение разошлось: метрики не конечны. Сбросьте эксперимент."
  ],
  [
    "Training diverged: weights are not finite.",
    "El entrenamiento divergió: pesos no finitos.",
    "Training divergiert: Gewichte sind nicht endlich.",
    "学習が発散しました。重みが有限値ではありません。",
    "L’entraînement a divergé : poids non finis.",
    "Обучение разошлось: веса не конечны."
  ],
  [
    "Seed must be an integer between 0 and 4294967295.",
    "La semilla debe ser un entero entre 0 y 4294967295.",
    "Seed muss eine ganze Zahl zwischen 0 und 4294967295 sein.",
    "シードは0から4294967295までの整数にしてください。",
    "La graine doit être un entier entre 0 et 4294967295.",
    "Seed должен быть целым числом от 0 до 4294967295."
  ],
  [
    "Epoch limit must be between 1 and 10000.",
    "El límite debe estar entre 1 y 10000 épocas.",
    "Das Epochenlimit muss zwischen 1 und 10000 liegen.",
    "エポック上限は1から10000までにしてください。",
    "La limite doit être comprise entre 1 et 10000 époques.",
    "Лимит эпох должен быть от 1 до 10000."
  ],
  [
    "Find the unique q ∈ {0,…,96} whose product bq leaves remainder a when divided by 97.",
    "Encuentra el único q ∈ {0,…,96} cuyo producto bq deja residuo a al dividirlo entre 97.",
    "Finde das eindeutige q ∈ {0,…,96}, für das das Produkt bq bei Division durch 97 den Rest a hat.",
    "bqを97で割った余りがaになる、一意な q ∈ {0,…,96} を求めます。",
    "Trouvez l’unique q ∈ {0,…,96} tel que le produit bq donne le reste a lorsqu’on le divise par 97.",
    "Найдите единственное q ∈ {0,…,96}, при котором остаток от деления произведения bq на 97 равен a."
  ],
  [
    "x mod 97 is the remainder from 0 to 96. In bq = a + 97k, k is an integer.",
    "x mod 97 es el residuo entre 0 y 96. En bq = a + 97k, k es un entero.",
    "x mod 97 ist der Rest von 0 bis 96. In bq = a + 97k ist k eine ganze Zahl.",
    "x mod 97 は0から96までの剰余です。bq = a + 97k のkは整数です。",
    "x mod 97 est le reste de 0 à 96. Dans bq = a + 97k, k est un entier.",
    "x mod 97 — остаток от 0 до 96. В равенстве bq = a + 97k число k целое."
  ],
  [
    "97 is prime, so every nonzero b has an inverse b⁻¹ satisfying b·b⁻¹ ≡ 1 (mod 97).",
    "97 es primo, por lo que todo b distinto de cero tiene un inverso b⁻¹ que cumple b·b⁻¹ ≡ 1 (mod 97).",
    "97 ist eine Primzahl. Daher besitzt jedes b ≠ 0 ein Inverses b⁻¹ mit b·b⁻¹ ≡ 1 (mod 97).",
    "97は素数なので、0でないすべてのbに b·b⁻¹ ≡ 1 (mod 97) を満たす逆元b⁻¹があります。",
    "97 est premier : tout b non nul possède un inverse b⁻¹ tel que b·b⁻¹ ≡ 1 (mod 97).",
    "97 — простое число, поэтому у каждого ненулевого b есть обратный элемент b⁻¹, для которого b·b⁻¹ ≡ 1 (mod 97)."
  ],
  [
    "Multiply a by that inverse, then take the remainder modulo 97.",
    "Multiplica a por ese inverso y toma el residuo módulo 97.",
    "Multipliziere a mit diesem Inversen und nimm den Rest modulo 97.",
    "aにその逆元を掛け、97を法とする剰余を求めます。",
    "Multipliez a par cet inverse, puis prenez le reste modulo 97.",
    "Умножьте a на обратный элемент и возьмите остаток по модулю 97."
  ],
  [
    "There are 97 × 96 = 9,312 equations. A fixed random split assigns 4,656 to training and 4,656 to testing on unseen pairs (a,b).",
    "Hay 97 × 96 = 9.312 ecuaciones. Una partición aleatoria fija asigna 4.656 al entrenamiento y 4.656 a la prueba con pares (a,b) no vistos.",
    "Es gibt 97 × 96 = 9.312 Gleichungen. Eine feste zufällige Aufteilung nutzt 4.656 zum Training und 4.656 zum Test auf unbekannten Paaren (a,b).",
    "式は97 × 96 = 9,312個です。固定したランダム分割で4,656個を学習に、未学習の組 (a,b) 4,656個をテストに使います。",
    "Il y a 97 × 96 = 9 312 équations. Une répartition aléatoire fixe en attribue 4 656 à l’entraînement et 4 656 au test sur des paires (a,b) inédites.",
    "Всего 97 × 96 = 9 312 уравнений. Фиксированное случайное разбиение отдаёт 4 656 в обучение и 4 656 в тест на ранее не встречавшихся парах (a,b)."
  ],
  [
    "For 10 / 3, the inverse of 3 is 65. The exact answer is 68, and multiplying it by 3 restores remainder 10.",
    "Para 10 / 3, el inverso de 3 es 65. La respuesta exacta es 68; al multiplicarla por 3 se obtiene de nuevo el residuo 10.",
    "Bei 10 / 3 ist das Inverse von 3 gleich 65. Die exakte Antwort ist 68; mal 3 ergibt sie wieder den Rest 10.",
    "10 / 3では、3の逆元は65です。正しい答えは68で、3を掛けると余りが10に戻ります。",
    "Pour 10 / 3, l’inverse de 3 est 65. La réponse exacte est 68 ; multipliée par 3, elle redonne le reste 10.",
    "Для 10 / 3 обратный элемент к 3 равен 65. Точный ответ — 68: после умножения на 3 остаток снова равен 10."
  ],
  [
    "The inverse rule generates the correct training labels. The network learns to predict q from examples of (a,b).",
    "La regla del inverso genera las respuestas correctas de entrenamiento. La red aprende a predecir q a partir de ejemplos de (a,b).",
    "Die Inversenregel erzeugt die richtigen Trainingsantworten. Das Netz lernt aus Beispielen von (a,b), q vorherzusagen.",
    "逆元の規則で正解ラベルを作ります。ネットワークは (a,b) の例からqを予測するように学習します。",
    "La règle de l’inverse produit les bonnes réponses d’entraînement. Le réseau apprend à prédire q à partir d’exemples de (a,b).",
    "Правило с обратным элементом задаёт правильные ответы для обучения. Сеть учится предсказывать q по примерам пар (a,b)."
  ],
  [
    "All examples use arithmetic modulo 97. The check multiplies the answer by b and recovers a as the remainder.",
    "Todos los ejemplos usan aritmética módulo 97. La comprobación multiplica la respuesta por b y recupera a como residuo.",
    "Alle Beispiele rechnen modulo 97. Zur Prüfung wird die Antwort mit b multipliziert; der Rest muss a sein.",
    "すべての例は97を法とする計算です。答えにbを掛け、その剰余がaになることを確認します。",
    "Tous les exemples utilisent l’arithmétique modulo 97. La vérification multiplie la réponse par b et retrouve a comme reste.",
    "Все примеры вычисляются по модулю 97. Для проверки ответ умножается на b: остаток должен совпасть с a."
  ],
  [
    "Rotate90 predicts one of 97 answer classes q from (a,b). One shared E embeds both operands and reads out the answer.",
    "Rotate90 predice una de 97 clases q a partir de (a,b). Una E compartida representa ambos operandos y produce la salida.",
    "Rotate90 sagt eine von 97 Antwortklassen q aus (a,b) voraus. Ein gemeinsames E bettet beide Operanden ein und liefert die Ausgabe.",
    "Rotate90は (a,b) から97クラスのqを予測します。共有Eが両方のオペランドを埋め込み、答えの出力にも使われます。",
    "Rotate90 prédit l’une des 97 classes q à partir de (a,b). Un E partagé représente les deux opérandes et lit la réponse.",
    "Rotate90 выбирает один из 97 классов q по (a,b). Общая E задаёт эмбеддинги обоих операндов и читает ответ."
  ],
  [
    "The current 1024-channel model has an 11–15-epoch memorization interval on five tested seeds, below the lab’s 25-epoch grokking criterion.",
    "El modelo actual de 1024 canales tiene un intervalo de memorización de 11–15 épocas en cinco semillas probadas, inferior al criterio de 25 épocas del laboratorio.",
    "Das aktuelle Modell mit 1024 Kanälen hat bei fünf getesteten Seeds eine Auswendiglernphase von 11–15 Epochen, unter dem Grokking-Kriterium von 25 Epochen.",
    "現在の1024チャネルモデルは5つのシードで11～15エポックの暗記期間を示しました。この実験のGrokking基準である25エポックには達していません。",
    "Le modèle actuel à 1024 canaux a une phase de mémorisation de 11 à 15 époques sur cinq graines testées, en dessous du critère de 25 époques du labo.",
    "У текущей модели на 1024 канала интервал запоминания составляет 11–15 эпох на пяти проверенных seed — меньше критерия grokking в 25 эпох."
  ],
  [
    "At least 25 continuous epochs with train ≥99% and held-out ≤10%, then a rise to ≥90% within 250 epochs and a later held-out ≥99% result. Every epoch is evaluated.",
    "Al menos 25 épocas consecutivas con entrenamiento ≥99% y prueba ≤10%, después un aumento a ≥90% en 250 épocas y finalmente prueba ≥99%. Se evalúa cada época.",
    "Mindestens 25 Epochen in Folge mit Training ≥99% und Test ≤10%, dann Anstieg auf ≥90% innerhalb von 250 Epochen und später Test ≥99%. Jede Epoche wird ausgewertet.",
    "学習正解率99%以上・テスト正解率10%以下が25エポック以上続き、250エポック以内に90%以上へ上昇し、その後99%以上に到達すること。毎エポック評価します。",
    "Au moins 25 époques consécutives avec entraînement ≥99% et test ≤10%, puis une montée à ≥90% en 250 époques et enfin un test ≥99%. Chaque époque est évaluée.",
    "Не менее 25 эпох подряд с обучением ≥99% и тестом ≤10%, затем рост до ≥90% за 250 эпох и позднее тест ≥99%. Оценка выполняется каждую эпоху."
  ],
  [
    "Seed 11, Rust/CUDA: held-out ≥99% at epoch 113. All five tested seeds reached ≥99% at epochs 105–113 without a memorization plateau. Browser arithmetic may change crossing epochs.",
    "Semilla 11, Rust/CUDA: prueba ≥99% en la época 113. Las cinco semillas probadas alcanzaron ≥99% entre las épocas 105 y 113 sin meseta de memorización. La aritmética del navegador puede cambiar las épocas de cruce.",
    "Seed 11, Rust/CUDA: Test ≥99% in Epoche 113. Alle fünf getesteten Seeds erreichten ≥99% in den Epochen 105–113 ohne Auswendiglernplateau. Browserarithmetik kann die Schwellenzeitpunkte verändern.",
    "シード11、Rust/CUDAでは113エポックでテスト99%以上。5つのシードすべてが暗記の停滞期なく105～113エポックで99%以上に到達しました。ブラウザーの計算では到達時期が変わることがあります。",
    "Graine 11, Rust/CUDA : test ≥99% à l’époque 113. Les cinq graines testées ont atteint ≥99% entre 105 et 113 époques, sans plateau de mémorisation. Le calcul du navigateur peut déplacer ces seuils.",
    "Seed 11, Rust/CUDA: тест ≥99% на эпохе 113. Все пять проверенных seed достигли ≥99% на эпохах 105–113 без плато запоминания. В браузере моменты пересечения порогов могут отличаться."
  ],
  [
    "Seed 11, Rust/CUDA: train ≥99% and held-out ≤10% at epochs 26–38; held-out ≥90% at 147 and ≥99% at 181. All five tested seeds reached ≥99% at epochs 178–188. Their 11–15-epoch plateaus are shorter than the 25-epoch grokking criterion. Beige: train ≥99%; violet: perfect train; held-out ≤10% in both.",
    "Semilla 11, Rust/CUDA: entrenamiento ≥99% y prueba ≤10% en las épocas 26–38; prueba ≥90% en 147 y ≥99% en 181. Las cinco semillas alcanzaron ≥99% entre 178 y 188. Sus mesetas de 11–15 épocas son inferiores al criterio de 25 épocas. Beige: entrenamiento ≥99%; violeta: entrenamiento perfecto; prueba ≤10% en ambos.",
    "Seed 11, Rust/CUDA: Training ≥99% und Test ≤10% in Epochen 26–38; Test ≥90% bei 147 und ≥99% bei 181. Alle fünf Seeds erreichten ≥99% in Epochen 178–188. Die Plateaus von 11–15 Epochen liegen unter dem 25-Epochen-Kriterium. Beige: Training ≥99%; violett: perfektes Training; jeweils Test ≤10%.",
    "シード11、Rust/CUDAでは26～38エポックで学習99%以上・テスト10%以下、147でテスト90%以上、181で99%以上。5つのシードすべてが178～188で99%以上に到達しました。11～15エポックの停滞期は25エポックの基準より短いものです。ベージュは学習99%以上、紫は学習100%で、両方ともテスト10%以下です。",
    "Graine 11, Rust/CUDA : entraînement ≥99% et test ≤10% aux époques 26–38 ; test ≥90% à 147 et ≥99% à 181. Les cinq graines ont atteint ≥99% entre 178 et 188. Leurs plateaux de 11 à 15 époques sont inférieurs au critère de 25 époques. Beige : entraînement ≥99% ; violet : entraînement parfait ; test ≤10% dans les deux cas.",
    "Seed 11, Rust/CUDA: обучение ≥99% и тест ≤10% на эпохах 26–38; тест ≥90% на 147 и ≥99% на 181. Все пять seed достигли ≥99% на эпохах 178–188. Плато 11–15 эпох короче критерия grokking в 25 эпох. Бежевый: обучение ≥99%; фиолетовый: обучение 100%; в обоих случаях тест ≤10%."
  ],
  [
    "Only width changes: 128 channels / 12,672 parameters, or 1024 / 101,376. Switching model tabs pauses the previous model and preserves its weights, optimizer and curves. Training starts from random initialization on your device.",
    "Solo cambia la anchura: 128 canales / 12.672 parámetros, o 1024 / 101.376. Al cambiar de modelo se pausa el anterior y se conservan sus pesos, optimizador y curvas. El entrenamiento empieza desde una inicialización aleatoria en tu dispositivo.",
    "Nur die Breite ändert sich: 128 Kanäle / 12.672 Parameter oder 1024 / 101.376. Ein Modellwechsel pausiert das vorherige Modell und behält Gewichte, Optimierer und Kurven. Das Training startet zufällig initialisiert auf deinem Gerät.",
    "変わるのは幅だけです。128チャネル・12,672パラメータ、または1024チャネル・101,376パラメータ。タブを切り替えると前のモデルを停止し、重み・最適化状態・曲線を保持します。端末上でランダム初期値から学習します。",
    "Seule la largeur change : 128 canaux / 12 672 paramètres, ou 1024 / 101 376. Changer de modèle met le précédent en pause et conserve ses poids, son optimiseur et ses courbes. L’entraînement part d’une initialisation aléatoire sur votre appareil.",
    "Меняется только ширина: 128 каналов / 12 672 параметра или 1024 / 101 376. При смене вкладки предыдущая модель ставится на паузу, её веса, оптимизатор и кривые сохраняются. Обучение начинается со случайной инициализации на вашем устройстве."
  ],
  [
    "All draws every E connection in its three shared roles and every Mesh edge. Pink is positive; blue is negative. Weights shows values; Change ×4 shows changes from initialization with a fixed scale. Highlight emphasizes a channel. Zoom, drag and Fit explore the graph; Channel expands the 2 → 1 Mesh and skip. Nodes show probe activations.",
    "Todos dibuja cada conexión de E en sus tres papeles compartidos y todas las conexiones de malla. Rosa es positivo; azul es negativo. Pesos muestra valores y Cambio ×4 los cambios desde el inicio con escala fija. Resaltar destaca un canal. Zoom, arrastre y Ajustar exploran el gráfico; Canal amplía la malla 2 → 1 y el residual. Los nodos muestran activaciones.",
    "Alle zeigt jede E-Verbindung in ihren drei gemeinsamen Rollen und alle Mesh-Kanten. Rosa ist positiv, Blau negativ. Gewichte zeigt Werte; Änderung ×4 zeigt Änderungen zur Initialisierung mit fester Skala. Hervorheben markiert einen Kanal. Zoom, Ziehen und Einpassen erkunden den Graphen; Kanal vergrößert das 2 → 1-Mesh und den Residualpfad. Knoten zeigen Aktivierungen.",
    "全体表示ではEの3つの共有用途の結合と全Mesh結合を描きます。正はピンク、負は青です。重みは値、変化×4は初期値からの差を固定目盛で表示します。強調でチャネルを選び、拡大・ドラッグ・全体表示で探索できます。チャネル表示は2 → 1 Meshと残差を拡大し、ノードはテスト式の活性値を示します。",
    "Tous affiche chaque connexion de E dans ses trois rôles partagés et toutes les connexions du maillage. Rose signifie positif, bleu négatif. Poids affiche les valeurs ; Variation ×4 les écarts à l’initialisation avec une échelle fixe. Surligner met un canal en évidence. Zoom, déplacement et Ajuster explorent le graphe ; Canal agrandit le maillage 2 → 1 et le résidu. Les nœuds montrent les activations.",
    "Все показывает каждую связь E в трёх общих ролях и все связи Mesh. Розовый — положительный вес, синий — отрицательный. Веса показывает значения; Изменение ×4 — разницу с инициализацией при фиксированном масштабе. Выделить отмечает канал. Масштаб, перетаскивание и Вместить помогают изучать граф; Канал раскрывает Mesh 2 → 1 и skip. Узлы показывают активации примера."
  ],
  [
    "|probe hidden꜀| × ‖centered Eᵀ꜀‖ measures a channel’s readout contribution. If channels share a pixel, its bar shows their maximum and selects that channel. The channel input selects any channel exactly.",
    "|hidden꜀ del ejemplo| × ‖Eᵀ꜀ centrada‖ mide la contribución del canal a la salida. Si varios canales comparten un píxel, la barra muestra su máximo y selecciona ese canal. El campo de canal permite elegir cualquiera exactamente.",
    "|hidden꜀ des Beispiels| × ‖zentriertes Eᵀ꜀‖ misst den Ausgabebeitrag eines Kanals. Teilen sich Kanäle ein Pixel, zeigt der Balken ihr Maximum und wählt diesen Kanal. Das Eingabefeld wählt jeden Kanal exakt.",
    "|テスト式のhidden꜀| × ‖中心化したEᵀ꜀‖ がチャネルの出力寄与を表します。複数チャネルが1画素を共有する場合、棒は最大値を示し、そのチャネルを選びます。数値入力で任意のチャネルを正確に選べます。",
    "|hidden꜀ de l’exemple| × ‖Eᵀ꜀ centré‖ mesure la contribution d’un canal à la sortie. Si plusieurs canaux partagent un pixel, la barre affiche leur maximum et sélectionne ce canal. Le champ numérique permet un choix exact.",
    "Вклад канала — модуль его значения в текущем примере, умноженный на длину вектора центрированных выходных весов. Если в одном столбце объединены несколько каналов, показан наибольший вклад. Поле «Канал» позволяет выбрать нужный индекс точно."
  ],
  [
    "Workspace view",
    "Vista del espacio de trabajo",
    "Arbeitsansicht",
    "作業画面の表示",
    "Vue de l’espace de travail",
    "Вид рабочей области"
  ],
  [
    "Chart metric",
    "Métrica del gráfico",
    "Diagrammkennzahl",
    "グラフの指標",
    "Mesure du graphique",
    "Метрика графика"
  ],
  [
    "Live training and held-out learning curves",
    "Curvas en tiempo real de entrenamiento y prueba",
    "Lernkurven für Training und Test in Echtzeit",
    "学習とテストのリアルタイム曲線",
    "Courbes d’entraînement et de test en direct",
    "Кривые обучения и теста в реальном времени"
  ],
  [
    "Network detail",
    "Detalle de la red",
    "Netzdetail",
    "ネットワークの詳細",
    "Détail du réseau",
    "Детализация сети"
  ],
  [
    "Network weight display",
    "Visualización de pesos de la red",
    "Anzeige der Netzgewichte",
    "ネットワークの重み表示",
    "Affichage des poids du réseau",
    "Отображение весов сети"
  ],
  [
    "Emphasize the selected channel without hiding any connections",
    "Resaltar el canal seleccionado manteniendo todas las conexiones",
    "Gewählten Kanal hervorheben und alle Verbindungen anzeigen",
    "すべての結合を表示したまま選択チャネルを強調",
    "Mettre le canal choisi en évidence en gardant toutes les connexions",
    "Выделить выбранный канал, сохранив все связи"
  ],
  [
    "Live weights of the complete model",
    "Pesos del modelo completo en tiempo real",
    "Alle Modellgewichte in Echtzeit",
    "モデル全体の重みをリアルタイム表示",
    "Poids du modèle complet en direct",
    "Все веса модели в реальном времени"
  ],
  [
    "Zoom network out",
    "Alejar la red",
    "Netz verkleinern",
    "ネットワークを縮小",
    "Réduire le réseau",
    "Уменьшить сеть"
  ],
  [
    "Fit all weights",
    "Ajustar todos los pesos",
    "Alle Gewichte einpassen",
    "すべての重みを表示",
    "Ajuster tous les poids",
    "Вместить все веса"
  ],
  [
    "Zoom network in",
    "Acercar la red",
    "Netz vergrößern",
    "ネットワークを拡大",
    "Agrandir le réseau",
    "Увеличить сеть"
  ],
  [
    "Selected mesh channel",
    "Canal de malla seleccionado",
    "Gewählter Mesh-Kanal",
    "選択したMeshチャネル",
    "Canal de maillage sélectionné",
    "Выбранный канал Mesh"
  ],
  [
    "Weight group",
    "Grupo de pesos",
    "Gewichtsgruppe",
    "重みのグループ",
    "Groupe de poids",
    "Группа весов"
  ],
  [
    "Center the readout over answer classes",
    "Centrar la salida sobre las clases de respuesta",
    "Ausgabe über Antwortklassen zentrieren",
    "答えのクラスに対して出力を中心化",
    "Centrer la sortie sur les classes de réponse",
    "Вычесть средний выходной вес по всем ответам"
  ],
  [
    "Raw weights or change from initialization",
    "Pesos originales o cambios desde la inicialización",
    "Gewichte oder Änderung zur Initialisierung",
    "重みの値または初期値からの変化",
    "Poids bruts ou variations depuis l’initialisation",
    "Значения весов или изменения с инициализации"
  ],
  [
    "Shared embedding weights for both operands and readout",
    "Pesos del embedding compartido de operandos y salida",
    "Gemeinsame Embedding-Gewichte für Operanden und Ausgabe",
    "両オペランドと出力で共有する埋め込みの重み",
    "Poids de l’embedding partagé par les opérandes et la sortie",
    "Общие веса для обоих входов и выходного преобразования"
  ],
  [
    "Shared embedding rotated by 90 degrees for operand b",
    "Embedding compartido rotado 90 grados para b",
    "Gemeinsames Embedding für b um 90 Grad gedreht",
    "オペランドb用に90度回転した共有埋め込み",
    "Embedding partagé tourné de 90 degrés pour b",
    "Общий эмбеддинг с поворотом на 90 градусов для b"
  ],
  [
    "Readout weights, centered over answer classes",
    "Pesos de salida centrados sobre clases de respuesta",
    "Ausgabegewichte, über Antwortklassen zentriert",
    "答えのクラスに対して中心化した出力の重み",
    "Poids de sortie centrés sur les classes de réponse",
    "Выходные веса с вычитанием среднего по всем ответам"
  ],
  [
    "Centered channel scales for all 64 channels",
    "Escalas centradas de los 64 canales",
    "Zentrierte Skalen aller 64 Kanäle",
    "全64チャネルの中心化した尺度",
    "Échelles centrées des 64 canaux",
    "Центрированные масштабы всех 64 каналов"
  ],
  [
    "Operand a",
    "Operando a",
    "Operand a",
    "オペランドa",
    "Opérande a",
    "Операнд a"
  ],
  [
    "Operand b",
    "Operando b",
    "Operand b",
    "オペランドb",
    "Opérande b",
    "Операнд b"
  ],
  [
    "Training controls",
    "Controles de entrenamiento",
    "Trainingssteuerung",
    "学習の操作",
    "Commandes d’entraînement",
    "Управление обучением"
  ],
  [
    "Training epoch progress",
    "Progreso de las épocas",
    "Epochenfortschritt",
    "学習エポックの進捗",
    "Progression des époques",
    "Прогресс эпох обучения"
  ],
  [
    "Close experiment information",
    "Cerrar información del experimento",
    "Experimentinformationen schließen",
    "実験の説明を閉じる",
    "Fermer les informations de l’expérience",
    "Закрыть описание эксперимента"
  ],
  [
    "{renderer} · {zoom}× · fixed scale",
    "{renderer} · {zoom}× · escala fija",
    "{renderer} · {zoom}× · feste Skala",
    "{renderer}・{zoom}×・固定目盛",
    "{renderer} · {zoom}× · échelle fixe",
    "{renderer} · {zoom}× · фиксированный масштаб"
  ],
  [
    "{title} weights",
    "Pesos de {title}",
    "Gewichte: {title}",
    "{title}の重み",
    "Poids : {title}",
    "Веса: {title}"
  ],
  [
    "Probe readout contributions for all {width} channels",
    "Contribuciones a la salida de los {width} canales",
    "Ausgabebeiträge aller {width} Kanäle",
    "全{width}チャネルの出力寄与",
    "Contributions des {width} canaux à la sortie",
    "Вклады всех {width} каналов в ответ на проверяемый пример"
  ],
  [
    "weights",
    "pesos",
    "Gewichte",
    "重み",
    "poids",
    "веса"
  ],
  [
    "weight changes from initialization",
    "cambios desde la inicialización",
    "Gewichtsänderungen zur Initialisierung",
    "初期値からの重みの変化",
    "variations depuis l’initialisation",
    "изменения весов с инициализации"
  ],
  [
    ", selected channel highlighted",
    ", canal seleccionado resaltado",
    ", gewählter Kanal hervorgehoben",
    "、選択チャネルを強調",
    ", canal choisi surligné",
    ", выбранный канал выделен"
  ],
  [
    ", no channel emphasis",
    ", sin resaltar un canal",
    ", ohne Kanalhervorhebung",
    "、チャネルの強調なし",
    ", sans surlignage de canal",
    ", без выделения канала"
  ],
  [
    "All: wheel to zoom, drag to pan. Arrow keys select a channel; 0 fits the network.",
    "Todos: rueda para zoom, arrastra para desplazar. Las flechas eligen un canal; 0 ajusta la red.",
    "Alle: Mausrad zum Zoomen, Ziehen zum Verschieben. Pfeile wählen einen Kanal; 0 passt das Netz ein.",
    "全体：ホイールで拡大、ドラッグで移動。矢印キーでチャネル選択、0で全体表示。",
    "Tous : molette pour zoomer, glisser pour déplacer. Les flèches choisissent un canal ; 0 ajuste le réseau.",
    "Все: колесо — масштаб, перетаскивание — сдвиг. Стрелки выбирают канал; 0 вмещает сеть."
  ],
  [
    "{node} · activation {value} · click to select",
    "{node} · activación {value} · clic para seleccionar",
    "{node} · Aktivierung {value} · zum Auswählen anklicken",
    "{node}・活性値 {value}・クリックして選択",
    "{node} · activation {value} · cliquez pour sélectionner",
    "{node} · активация {value} · нажмите для выбора"
  ],
  [
    "Channel {channel}",
    "Canal {channel}",
    "Kanal {channel}",
    "チャネル{channel}",
    "Canal {channel}",
    "Канал {channel}"
  ],
  [
    "Final RMSNorm h[{channel}]",
    "RMSNorm final h[{channel}]",
    "Letztes RMSNorm h[{channel}]",
    "最終RMSNorm h[{channel}]",
    "RMSNorm final h[{channel}]",
    "Финальная RMSNorm h[{channel}]"
  ],
  [
    "Layer {layer} · {from} → {to} · weight {value}",
    "Capa {layer} · {from} → {to} · peso {value}",
    "Schicht {layer} · {from} → {to} · Gewicht {value}",
    "層{layer}・{from} → {to}・重み {value}",
    "Couche {layer} · {from} → {to} · poids {value}",
    "Слой {layer} · {from} → {to} · вес {value}"
  ],
  [
    "Channel {channel} · {kind} {residue} · {weightKind} {value}",
    "Canal {channel} · {kind} {residue} · {weightKind} {value}",
    "Kanal {channel} · {kind} {residue} · {weightKind} {value}",
    "チャネル{channel}・{kind} {residue}・{weightKind} {value}",
    "Canal {channel} · {kind} {residue} · {weightKind} {value}",
    "Канал {channel} · {kind} {residue} · {weightKind} {value}"
  ],
  [
    "Pink: wrong answer ahead. Green: correct answer ahead. Each column is a real snapshot; epoch labels identify its time.",
    "Rosa: gana una respuesta incorrecta. Verde: gana la correcta. Cada columna es una captura real y su época indica el momento.",
    "Rosa: falsche Antwort liegt vorn. Grün: richtige Antwort liegt vorn. Jede Spalte ist eine echte Aufnahme mit Epochenangabe.",
    "ピンク：不正解の答えが優勢。緑：正解が優勢。各列は実際の記録で、エポック表示が時点を示します。",
    "Rose : une mauvaise réponse est devant. Vert : la bonne est devant. Chaque colonne est une vraie capture avec son époque.",
    "Розовый: впереди ошибочный ответ. Зелёный: впереди правильный. Каждая колонка — реальный снимок, время указано в эпохах."
  ],
  [
    "RoFormer: pairwise rotations ↗",
    "RoFormer: rotaciones por pares ↗",
    "RoFormer: paarweise Rotationen ↗",
    "RoFormer：対ごとの回転 ↗",
    "RoFormer : rotations par paires ↗",
    "RoFormer: попарные повороты ↗"
  ],
  [
    "Shared E in three roles + Dense A",
    "E compartida en tres funciones + densa A",
    "Gemeinsames E in drei Rollen + dichte A",
    "共有Eの3つの役割＋全結合A",
    "E partagé dans trois rôles + dense A",
    "Общая E в трёх ролях + плотная A"
  ],
  [
    "Dense matrix weights",
    "Pesos de la matriz densa",
    "Gewichte der dichten Matrix",
    "全結合行列の重み",
    "Poids de la matrice dense",
    "Веса плотной матрицы"
  ],
  [
    "Selected channel",
    "Canal seleccionado",
    "Ausgewählter Kanal",
    "選択したチャネル",
    "Canal sélectionné",
    "Выбранный канал"
  ],
  [
    "Probe readout contributions",
    "Contribuciones de salida",
    "Ausgabebeiträge der Probe",
    "試行入力の出力への寄与",
    "Contributions de sortie du test",
    "Вклад каналов в ответ для примера"
  ],
  [
    "Grokking is late generalization after memorizing the training examples. Watch the held-out accuracy to see when unseen pairs become correct.",
    "Grokking es la generalización tardía tras memorizar los ejemplos de entrenamiento. Observa la precisión de prueba para ver cuándo se resuelven pares nuevos.",
    "Grokking ist spätes Generalisieren nach dem Auswendiglernen der Trainingsbeispiele. Die Testgenauigkeit zeigt, wann unbekannte Paare korrekt werden.",
    "Grokkingは学習例の暗記後に起こる遅い汎化です。テスト正解率で未知の組が正しくなる時期を確認できます。",
    "Le grokking est une généralisation tardive après la mémorisation des exemples. L’exactitude de test montre quand les paires inédites deviennent correctes.",
    "Grokking — позднее обобщение после запоминания обучающих примеров. Точность на тесте показывает, когда модель начинает правильно решать незнакомые пары."
  ],
  ["{count} threads","{count} hilos","{count} Threads","{count}スレッド","{count} threads","Потоков: {count}"],
  ["SIMD · 1 thread","SIMD · 1 hilo","SIMD · 1 Thread","SIMD・1スレッド","SIMD · 1 thread","SIMD · 1 поток"],
  ["SIMD · 2 threads","SIMD · 2 hilos","SIMD · 2 Threads","SIMD・2スレッド","SIMD · 2 threads","SIMD · 2 потока"],
  ["SIMD · 4 threads","SIMD · 4 hilos","SIMD · 4 Threads","SIMD・4スレッド","SIMD · 4 threads","SIMD · 4 потока"],
  ["Scalar · 1 thread","Escalar · 1 hilo","Skalar · 1 Thread","スカラー・1スレッド","Scalaire · 1 thread","Скалярный · 1 поток"],
  ["Scalar · 2 threads","Escalar · 2 hilos","Skalar · 2 Threads","スカラー・2スレッド","Scalaire · 2 threads","Скалярный · 2 потока"],
  ["Scalar · 4 threads","Escalar · 4 hilos","Skalar · 4 Threads","スカラー・4スレッド","Scalaire · 4 threads","Скалярный · 4 потока"],
  [
    "Shared memory is unavailable; using one CPU thread.",
    "La memoria compartida no está disponible; se usa un hilo de CPU.",
    "Gemeinsamer Speicher ist nicht verfügbar; ein CPU-Thread wird verwendet.",
    "共有メモリーを利用できないため、CPUスレッドを1つ使います。",
    "La mémoire partagée est indisponible ; un seul thread CPU est utilisé.",
    "Общая память недоступна; используется один поток CPU."
  ],
  [
    "Parallel WASM could not start; using one CPU thread. {error}",
    "No se pudo iniciar WASM paralelo; se usa un hilo de CPU. {error}",
    "Paralleles WASM konnte nicht starten; ein CPU-Thread wird verwendet. {error}",
    "並列WASMを起動できないため、CPUスレッドを1つ使います。{error}",
    "Le WASM parallèle n’a pas pu démarrer ; un seul thread CPU est utilisé. {error}",
    "Параллельный WASM не запустился; используется один поток CPU. {error}"
  ],
];
