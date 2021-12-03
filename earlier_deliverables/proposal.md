### Art by the Emotion ###

**Dataset**
There is an annotaed dataset of emotions evoked by around 4000 artworks (mostly paintings) taken from WikiArt. 

The dataset is available here: 
[WikiArt Emotions Dataset](http://saifmohammad.com/WebPages/wikiartemotions.html#ethics)

**Questions**
The model built should correctly identify the emotions felt by the annotators after training on the training set of images. 

The dataset (csv, of emotions) as well as specifying the emotions felt by the annotator also specifies whether the painting includes a depiction of a face 
(so a straighforward portrait or not) and also whether the annotator liked the painting or not. 

Once trained and able to identify these emotional networks, further work could include feeding other images to the model (there are other WikiArt image datasets out there)
& have it group paintings and also certain art periods by emotion. ie. if a certain emotion comes up more in a certain art period. 

**Tools** 
- Pandas, Numpy for EDA 
- Keras for modelling 

**MVP Goal**
Have a model running tuned and able to identify the emotions in the dataset. 
