# Project-phpjj1
This is the ADA 2024/25 project repository for the group phpjj1.

# How Movies are Built: The Emotional Landscape in Cinema

## Abstract
In cinema, emotions are not merely a component of storytelling but the foundation upon which entire films are constructed. This project aims to analyze how movies are built based on different emotions across genres and how these emotions have evolved over time. Additionally, it seeks to uncover relationships between emotional tones in movies and factors such as language, as well as the age and gender of involved actors. To fully understand the impact of emotions in films, we will also explore how these emotions influence a movie’s success, as measured by user ratings.

Our project primarily utilizes the CMU movie data set, which includes movie plot summaries sourced from Wikipedia, that are used to extract the emotions of a film. This data is enriched with an external dataset containing user ratings. Through this analysis, we aim to show how leveraging emotional narratives can be used to produce the next blockbuster.

## Research Questions
- What are the predominant emotional tones used in films and how do they vary across different movie genres?  
- How has the emotional tone of films evolved over time within each genre?  
- How does the emotional tone of films influence the selection of actor traits, such as age and gender?  
- Does the emotional tone of films differ between movies in different languages?  
- Can films be clustered based on their emotional tone, and do these clusters reveal distinct patterns in consumer ratings?

By answering these questions, we hope to provide insights into how emotional tones shape cinematic experiences, influence casting decisions, and connect with audiences globally.

## Data
From the provided CMU Movie Summary Corpus, we use the movie metadata, the character metadata, and the movie plot summaries. We decided to merge them by a left join to initially not lose any data and handle resulting missing values by grouping. We explored each of the three datasets with initial exploratory data analysis to find missing values and generate relevant plots.

We decided to use user ratings to measure the success of a movie instead of financial revenue, because the latter can be highly influenced by good marketing and famous actors and therefore is not the best indicator of whether people truly liked the movie. In our main dataset, there are no user ratings, so we decided to merge it with the [MovieLens dataset](https://grouplens.org/datasets/movielens/32m/). This dataset contains 32 million movie ratings for 87,585 movies. We merge the datasets based on a cleaned title and the release year of the movie.

## Methods

### Natural Language Processing
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
The impact of emotions on the actor traits age and gender is analyzed using regression analysis. This method assesses relationships between the emotional tones of films and the age and gender of involved actors. Additionally, regression analysis will be used to explore connections between movie languages and emotional tones.

### Clustering
To explore the connection between emotional tones and user ratings, we will cluster movies based on their emotional profiles. Patterns between clusters and user ratings are examined to determine how the different emotions influence user ratings and how the influence may vary across genres.

## Timeline

### 15.11.2024 - 22.11.2024
- Homework
- Extraction of emotions for all movie plots

### 22.11.2024 - 29.11.2024  
- Comparison of emotions across genres, including visualizations  
- Aggregate emotions by release date  
- Perform time series analysis on aggregated data  

### 29.11.2024 - 06.12.2024
- Regression analysis between emotional tones and the actor traits gender and age
- Regression analysis between emotional tones and the language of a movie
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
- **Filippo**: Time Series Analysis and complete emotion detection 
- **Freeman**: Clustering of emotions and data story
- **Julian**: Regression analysis and clustering
- **Sean**: Emotion aggregation and visualizations
- **Levin**: Regression analysis and data story

## Questions for the TA
N/A


## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>
# Necessary if using a conda environment otherwise pip will install packages globally even if the conda environment is active
conda install pip


# install requirements
pip install -r pip_requirements.txt
```

### How to use the library
To use the code, the data mentioned above in the data section has to be downloaded. Make sure to place all data in the root data folder (for details look below) and put all datafiles from the CMU Movie dataset into a folder named `MovieSummaries`. For the MovieLens dataset put all files in a folder named `ml-32m`. To see our inital analysis and data pipeline create a conda environment as described above and run all cells of our `result.ipynb` on the source level as shown below.



## Project Structure

The directory structure of our project looks like this:

```
├── data                        <- Project data files
│   ├── ml-32m                            <- Contains the MovieLens data
│   ├── MovieSummaries                    <- Contains the CMU Movie data
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- Our result notebook
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Downloading Raw Datasets

**Primary Dataset**: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) (`MovieSummaries`)

#### Secondary Datasets:
- [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/#:~:text=for%20new%20research-,MovieLens%2032M,-MovieLens%2032M%20movie) (`ml-32m`)


For each dataset, download the zip file and place it in the root level `data` directory, as described above.
