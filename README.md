## Environment Design: Predicting Emotions an Artwork evokes

**Workflow**

[Notebook1: Scraping of Images](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/1_scrape_images.ipynb) 

[Notebook2: EDA of the different datasets](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/2_data_organization_EDA.ipynb) 

[Notebook3_t30: Multiclass multilabel, using dataset with 30% threshold](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/3_t30_model_20_class_multilabel.ipynb) 

[Notebook3_t50: Multiclass multilabel, using dataset with 50% threshold](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/3_t50_20_class_multilabel.ipynb) 

[Notebook4: Multiclass single label](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/4_20_class_single_label.ipynb) 

[Notebook5: Binary with happiness and surprise as the 2 classes](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/5_binary_happiness_surprise.ipynb) 

[Notebook6: Binary with positive and negative as the 2 classes](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/6_binary_positive_neg.ipynb) 

[Notebook7: Three classes with positive, mixed and negative emotions](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/7_three_classes.ipynb) 

In case this last notebook doesn't render, as it is a really large file; here are 2 files: The first one is the building of the model, the second one is the code for the image prediction/ evaluation. The results of these predictions can mostly be found in the presentation. 

[Notebook7a: 3-class modeling](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/7a_three_classes_modeling.ipynb) 

[Notebook7b: 3-class predicting](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/7b_three_classes_predictions.ipynb) 

[Presentation](https://github.com/zey-o/DeepLearning_Wikiart_Emotions/blob/main/Presentation_DL_Wikiart_Emotions.pdf) 

**Abstract**

This project uses the datasets from a psychological study by Saif M. Mohammad and Svetlana Kiritchenko in which 20 annotators were shown over 4000 artworks and asked to write down the emotions the artworks evoked. The images from the dataset were scraped from WikiArt website, which is a rich database of artworks. Several deep learning models were built in order to predict the emotion evoked by the artwork. The binary classification model for whether the image evokes positive or negative emotions achieved 71% accuracy, while the tri-class model which is closer to the original study in which the authors divide the emotions into mixed, positive and negative classes achieved a 68% accuracy. The project was proposed to be put to a business use case to be deployed for an environment design by use of certain images, specifically the design of a film set. 


**Design**

The goal of this project was to build a neural network that would predict the emotion evoked by an artwork in order to then use other images in the design of physical environments (specifically film sets). 

I started off trying to build a multi-label, multi-class model (20 emotions) in which an artwork could be associated with more than one emotion. I tried various models for the three different datasets I had (explained below) and the ones that achieved the best accuracies were 1. One with transfer learning (MobileNet) with Dropouts and L2 and EarlyStopping 2. One simple single dense layer 3. One where I did use the VGG16 architecture. The accuracies oscillated between ~25 - 30% and the best one was achieved using the dataset where more emotions were marked as present (30% threshold). 

Then I went on to pose a binary classification problem. The first model I built was classifying the top 2 emotions in the artworks: Happiness vs. Surprise. The model was able to detect with a 85% accuracy. But this problem didn’t present a good business use case. I then went on to build a model for the Negative vs. Positive Emotion evoked by an artwork. I was able to get the best accuracy, a 71%, using transfer learning (MobileNet) architecture with 2 simple Dense layers. Then I added a third class, Mixed Emotions, which was loyal to the grouping proposed by the authors of the study. In this problem, the same model architecture as the binary problem achieved the highest accuracy of 68%. There was a slight class imbalance which was corrected for during training. 

We are assuming in this modeling that only one emotion (and a very generalized one: positive/negative/mixed) is evoked by the artwork shown. Further data would need to be collected as well as further classes incorporated for the model to be closer to deployment. 

**Data**

The dataset(s) can be found here: [WikiArt Emotions Dataset](http://saifmohammad.com/WebPages/wikiartemotions.html#ethics)
1 of the datasets lists the URL links for the images used. The links were used to scrape the images from the WikiArt website. 
1 of the datasets has the probability of each emotion being evoked by each artwork. 
The other 3 datasets are one hot encoded to show whether an emotion was evoked by an artwork or not. Consecutively, 30, 40 or 50 % of the annotators would have to have noted an emotion down for an emotion to be marked as evoked by an artwork. 

**Algorithms**
- flow_from_dataframe / ImageGenerator for image processing 
- Transfer Learning with: VGG16, MobileNet 
- Data Augmentation
- Dropout + L2 Regularization + EarlyStopping + ReduceLROnPlateau for dealing with overfitting 

**Tools**
- Python: Pandas & Numpy for EDA
- OS, Pathlib, OpenCV for directory
- Matplotlib, Seaborn for Visualization 
- Tensorflow/ Keras for modeling 
- Google Colab for cloud computing

**Communication**

The project was presented with a slideshow, and can be found as a presentation PDF. 

