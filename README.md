# How Movies are Built: The Emotional Landscape in Cinema

## Abstract
In cinema, emotions are not merely a component of storytelling but the foundation upon which entire films are constructed. This project aims to analyze how movies are built based on different emotions across genres and how these emotions have evolved over time. Additionally, it seeks to uncover relationships between the emotional profiles of movies and factors such as language, as well as the age and gender of the actors involved. To fully understand the impact of emotions in films, we will also explore how these emotions influence a movie’s success, as measured by user ratings.

Our project primarily utilizes the CMU movie data set, which includes movie plot summaries sourced from Wikipedia, that are used to extract the emotions of a film. This data is enriched with an external dataset containing user ratings. Through this analysis, we aim to show how leveraging emotional narratives can be used to produce the next blockbuster.

## Research Questions
- What are the predominant emotional tones used in films and how do they vary across different movie genres?  
- How has the overall tone of films evolved over time within each genre?  
- How does the emotional tone of films influence the selection of actor traits, such as age and gender?  
- Does the emotional tone of films differ between movies in different languages?  
- Can films be clustered based on their emotional tone, and do these clusters reveal distinct patterns in consumer ratings?

By answering these questions, we hope to provide insights into how emotional tones shape cinematic experiences, influence casting decisions, and connect with audiences globally.

## Data
<!-- From the provided CMU Movie Summary Corpus, we use the movie metadata, the character metadata, and the movie plot summaries. We decided to merge them by a left join to initially not lose any data and handle resulting missing values by grouping. We explored each of the three datasets with initial exploratory data analysis to find missing values and distributions of important columns. -->
From the provided CMU Movie Summary Corpus, we use the movie metadata, the character metadata, and the movie plot summaries. We decided to merge them by a left join to initially not lose any data and handle resulting missing values by grouping. We explored each of the three datasets with initial exploratory data analysis to find missing values and generate relevant plots.

We decided to use user ratings to measure the success of a movie instead of financial revenue, because the latter can be highly influenced by good marketing and famous actors and therefore is not the best indicator of whether people truly liked the movie. In our main dataset, there are no user ratings, so we decided to merge it with the [MovieLens dataset](https://grouplens.org/datasets/movielens/32m/). This dataset contains 32 million movie ratings for 87,585 movies. We merge the datasets based on a cleaned title and the release year of the movie.

## Methods

### Natural Language Processing
<!-- To uncover how films use emotions and analyze their evolution and impact, we first detect emotions from the movie plot summaries sourced from Wikipedia. The movie metadata is merged with character metadata and plot summaries to create a unified dataset. Emotions are then extracted using the pre-trained [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) model from Hugging Face. It predicts Ekman’s 6 basic emotions as well as a neutral class. -->
To uncover how films use emotions and analyze their evolution and impact, we first detect emotions from movie plot summaries sourced from Wikipedia. The movie metadata is merged with character metadata and plot summaries to create a unified dataset. Emotions are then extracted using various pre-trained models:
- **[j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)**:  
  This model predicts Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise) along with a neutral class, making it ideal for capturing foundational emotional elements in movie plots.

- **[SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)**:  
  Predicts 28 emotions, such as range of emotions like amusement, excitement, curiosity, and joy. This allows us to analyze a broader spectrum of emotional expression in film narratives.

- **NRCLex** ([source](https://pypi.org/project/NRCLex/)):  
  Predicts fear, anger, anticipation, trust, surprise, positive/negative sentiment, sadness, disgust, and joy.

For Milestone 2, we will evaluate each model's performance to determine the most suitable approach for our dataset. Additionally, for now in Milestone 2, we are going to run models on subset of plot summaries due to high computational cost. We will then choose the final model based on evaluations and run it on the whole summary plot data using external resources for Milestone 3.

### Aggregation and Comparison
To explore the connections between genres and emotions, the extracted emotional data is aggregated at the genre level. The resulting emotional profiles are then compared across genres to identify differences and patterns.

### Time Series Analysis
To examine how the use of emotions has evolved across genres over time, we will perform a time series analysis. Emotional scores for each genre will be aggregated by release year, enabling us to identify trends and shifts in emotional tones over the decades.

### Regression Analysis
The impact of emotions on the actor traits age and gender is analyzed using regression analysis. This method assesses relationships between the emotional tones of films and the age and gender of involved actors. Additionally, regression analysis may be used to explore connections between movie languages and emotional tones, complementing the chi-squared test described below.

### Chi-Squared Test
To investigate the relationship between movie language and emotional tones, a chi-square test will be conducted. This test will evaluate whether the distribution of emotions differs significantly across languages, under the null hypothesis that language has no impact on emotional tone. Based on the results, regression analysis may be employed for visualization and further exploration.

### Clustering
To explore the connection between emotional tones and user ratings, we will cluster movies based on their emotional profiles. Patterns between clusters and user ratings are examined to determine how the different emotions influence user ratings and how the influence may vary across genres.

## Timeline

### 15.11.2024 - 22.11.2024
- Homework

### 22.11.2024 - 29.11.2024
- Creation of emotional profiles for each movie through the extracted emotions  
- Comparison of profiles across genres, including visualizations  
- Aggregate emotions by release date  
- Perform time series analysis on aggregated data  

### 29.11.2024 - 06.12.2024
- Regression analysis between emotional profiles and the actor traits gender and age  
- Chi-squared test between different emotions and movie language  
- Visualization of results  

### 06.12.2024 - 13.12.2024
- Conceptualize the data story  
- Create blog post  
- Clustering of emotions and examination of connection to user ratings  
- Visualization of the cluster analysis results  

### 13.12.2024 - 20.12.2024
- Write the data story as blog post  
- Create missing visualizations  
- Code and repository cleaning  

## Organization of Tasks Across the Team
- **Fillipo**:  
- **Freeman**:  
- **Julian**:  
- **Sean**:  
- **Levin**:  

## Questions for the TA
- N/A
