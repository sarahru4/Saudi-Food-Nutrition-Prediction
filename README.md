# Saudi Food Nutrition Prediction

A deep learning project that estimates the nutritional values of Saudi and Arabic dishes from a single food image. Given a photo, the model predicts four values at once: **calories, protein, fat, and carbohydrates** (multi-output regression).

Global food databases underrepresent traditional Arabic meals, which makes it hard to find reliable nutrition information for dishes like kabsa or jareesh. This project builds a dedicated Arabic food dataset from scratch and uses it to train image-based nutrition models.

Developed as the IT362 Data Science course project at King Saud University.

---

## Objectives

1. Build a multi-output regression model that predicts calories, protein, fat, and carbohydrates from a food image.
2. Identify which Arabic dishes are highest and lowest in calories.
3. Identify which dishes are the most and least protein-dense (protein per 100 kcal).
4. Provide a clear nutritional overview of traditional Arabic foods.

---

## Repository Contents

| File | Description |
|---|---|
| `IT362_Project.ipynb` | Full pipeline: data collection, cleaning, EDA, and modeling |
| `it362project_reportt.pdf` | Project report |
| `Poster.pdf` | Project poster |

---

## Pipeline

### Phase 1: Data Collection

The primary dataset was built by web scraping three sources with `requests` and `BeautifulSoup`:

| Source | Content | Rows |
|---|---|---|
| [Najd Village](https://najdvillage.com/menu/?lang=en) | Restaurant menu (Saudi dishes) | 45 |
| [Puck Arabia](https://www.puckarabia.com/en/recipes/categories/arabic-recipes/) | Arabic recipes | 94 |
| [butfirstchai](https://butfirstchai.com/category/middle-eastern-recipes/) | Middle Eastern recipes | 68 |

For each dish, the scrapers collected the name, description, recipe, and image. Missing nutrition values were filled using the **Nutritionix API**, which estimates calories and macronutrients from the dish name and ingredients. The three sources were then merged into one dataset of 207 dishes.

### Phase 2: Cleaning, Image Processing, and EDA

**Data cleaning**
- Removed rows where all four nutrition values were zero (scraping noise)
- Checked for duplicates (none found)
- Consolidated images from all sources into a single folder and updated the paths

**Image processing**
- Converted all images to 3-channel RGB
- Resized to 224 x 224 pixels
- Augmented the data with horizontal and vertical flips, which tripled the dataset from 205 to 615 images

**Secondary dataset**
- [NutritionVerse-Real](https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real) from Kaggle (real food images with measured nutrition values)
- 225 dishes kept after removing rows without images

**Exploratory Data Analysis**
- Descriptive statistics, histograms, and boxplots for each nutrient
- Correlation analysis: calories are most strongly correlated with fat (r = 0.95) and carbohydrates (r = 0.91)
- Top and bottom 10 dishes by calories
- Top and bottom 10 dishes by protein density
- Comparison between the primary and secondary datasets

### Phase 3: Modeling

**Data split**
- Combined the primary and secondary datasets (840 images)
- Grouped augmented images by dish, so all versions of the same dish land in the same split and nothing leaks from train to test
- Used **K-Means clustering** on nutrition values to create 4 nutritional strata
- Split 80/20 with `StratifiedShuffleSplit` on these strata: 660 training images, 180 test images

**Models**

Three pretrained CNNs (ImageNet weights) were used with transfer learning. Each backbone was frozen, and a regression head was added on top:

```
Backbone (frozen) -> GlobalAveragePooling2D -> Dense(ReLU) -> Dropout -> Dense(4)
```

| Model | Backbone | Trainable Params |
|---|---|---|
| Model 1 | MobileNetV2 | 164K |
| Model 2 | EfficientNetB0 | 329K |
| Model 3 | ResNet50V2 | 526K |

All models were trained for 30 epochs with the Adam optimizer, MSE loss, and MAE as a metric.

---

## Results

Per-target MAE on the test set:

| Model | Protein (g) | Calories (kcal) | Fat (g) | Carbs (g) | Mean R² |
|---|---|---|---|---|---|
| MobileNetV2 | 30.16 | 527.10 | **28.88** | **57.90** | 0.084 |
| EfficientNetB0 | 32.09 | 608.01 | 31.57 | 64.89 | -0.003 |
| **ResNet50V2** | **26.34** | **522.13** | 29.51 | 62.98 | **0.111** |

**ResNet50V2 performed best overall.** It had the lowest error for protein and calories and the highest mean R². EfficientNetB0 failed to learn meaningful patterns (R² near zero).

The R² values are low for all three models. Estimating nutrition from a single image is a hard task, especially since portion size is hard to judge from a photo, and the dataset is small. Possible improvements include fine-tuning the backbone layers, collecting more images per dish, and adding portion size information.

---

## Key EDA Insights

- **Highest-calorie dishes:** rich desserts and cheese-based pastries such as Umm Ali, Mutabbaq, and Fried Qatayef with Cream
- **Lowest-calorie items:** spice mixes and light sides such as Bezar, Gulf Baharat, and Green Salad
- **Most protein-dense dishes:** lean, meat-based dishes such as Meat Sambosa, Chicken Steam, Liver, Chicken Kabsah, and Chicken Saleek (about 10 to 12 g of protein per 100 kcal)
- **Least protein-dense dishes:** desserts and sweet drinks such as Qamar al-Din and Kunafa

---

## Tech Stack

- **Language:** Python
- **Data collection:** requests, BeautifulSoup, Nutritionix API, Kaggle API
- **Data processing:** pandas, NumPy, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (K-Means, StratifiedShuffleSplit, metrics)
- **Deep learning:** TensorFlow / Keras (MobileNetV2, EfficientNetB0, ResNet50V2)
- **Environment:** Google Colab

---

## How to Run

1. Open `IT362_Project.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Add your own API keys:
   - A **Nutritionix** App ID and API Key for the nutrition-filling step
   - A **Kaggle** `kaggle.json` file to download the NutritionVerse-Real dataset
3. Run the cells in order. A GPU runtime is recommended for the modeling phase.

---

## Dataset Biases

- **Representation:** Gulf dishes are overrepresented, while North African and some Levantine cuisines are underrepresented.
- **Measurement:** Nutrition values come from menus, recipe websites, and an API, not laboratory measurements, so portion sizes and preparation methods vary.
- **Popularity:** The dataset favors common dishes (such as kabsa and mandi) over rare regional dishes.
