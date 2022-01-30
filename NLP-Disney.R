library(utf8)
library(ggplot2)
library(tidyverse)
library(viridis)
library(quanteda)
require(quanteda.textmodels)
library("quanteda.textplots")
library(udpipe)
library(lattice)
require(caret)

###PROBLEMA 1

##Leemos el dataset (formato UTF-8) y comporbamos el número de filas

dr <- read.csv("DisneyLandReviews.csv", encoding = "UTF-8" )

nrow(dr)

##Comprobamos valores NULL

sapply(dr, function(x) sum(is.na(x)))

##Comprobamos reviews duplicadas

dim(dr[duplicated(dr$Review_Text),])[1]

##Eliminar reviews duplicadas y comprobar que se han eliminado correctamente

dr <- dr[!duplicated(dr$Review_Text), ]

dim(dr[duplicated(dr$Review_Text),])[1]

nrow(dr)

## Eliminamos DisneyLand_ de Branch

dr$Branch = gsub("^.*_","", dr$Branch)

## Notas de las opiniones de HongKong

dr_HK <- dr[dr$Branch == "HongKong",]

p <- ggplot(dr_HK, aes(Rating))  +
  geom_bar() + geom_text(stat='count', aes(label = scales::percent(..prop..),
                                           y= ..prop.. ), vjust=1)

p + labs(x= "Puntuación", y="Total opiniones", 
         title="Opniniones/Puntuación HongKong", subtitle = "Opiniones: 9606")

## Notas de las opiniones de California

dr_C <- dr[dr$Branch == "California",]

p <- ggplot(dr_C, aes(Rating))  +
  geom_bar() + geom_text(stat='count', aes(label = scales::percent(..prop..),
                                           y= ..prop.. ), vjust=1)

p + labs(x= "Puntuación", y="Total opiniones", 
         title="Opniniones/Puntuación California", subtitle = "Opiniones: 19399")

## Notas de las opiniones de París


dr_P <- dr[dr$Branch == "Paris",]

p <- ggplot(dr_P, aes(Rating))  +
  geom_bar() + geom_text(stat='count', aes(label = scales::percent(..prop..),
                                           y= ..prop.. ), vjust=1)

p + labs(x= "Puntuación", y="Total opiniones", 
         title="Opniniones/Puntuación París", subtitle = "Opiniones: 13627")

## Evolución de las reviews / años


# Reviews años

Disney_Date <- dr %>%
  mutate(Date = as.Date(paste(Year_Month, "-01", sep = ""))) %>%
  mutate(Date = as.POSIXct(Date))%>%
  mutate(Date = format(Date, "%Y")) %>%
  count(Date, Branch)

#Eliminamos filas que Date = NA y creamos el gráfico

Disney_Date <- na.omit(Disney_Date)

ggplot(Disney_Date , aes(x = Date, y = n,group = Branch ,color = Branch))+
  geom_line()+
  theme(legend.position = "bottom")+
  scale_color_viridis(discrete = TRUE) +
  ylab("Total opiniones") +
  xlab("Años") +
  ggtitle("Evolución del numero de opiniones (2010 ~ 2019) / Parque")




### PROBLEMA 2

## Subset Reviews positivas (puntuación >= 4)

dr_Positive <- dr[dr$Rating >= 4,]

## Subset Reviews negativas (puntuación < 4)

dr_Negative <- dr[dr$Rating < 4,]

## Palabras mas comunes (POSITIVAS)

# Stopwords extras
mystopwords <- c("park", "disneyland", "disney", "get", "one", "go", "just")

# Data cleaning
corpus_pos <- dfm(tokens_remove(tokens(dr_Positive$Review_Text,
                                       remove_punct = TRUE,
                                       remove_numbers = TRUE, 
                                       remove_symbols = TRUE), 
                                pattern = c(stopwords(), mystopwords)))

topfeatures(corpus_pos)

textplot_wordcloud(corpus_pos, color = "green")


## Palabras mas comunes (NEGATIVAS)

corpus_pos <- dfm(tokens_remove(tokens(dr_Negative$Review_Text,
                                       remove_punct = TRUE,
                                       remove_numbers = TRUE, 
                                       remove_symbols = TRUE), 
                                pattern = c(stopwords(), mystopwords)))

topfeatures(corpus_pos)

textplot_wordcloud(corpus_pos, color = "red")


## Análisi profunfo reviews negativas (RAKE)

#Muestra de 1000 opinones de las 8731 possibles (Una major muestra tarda demasiado tiempo)
test <- dr_Negative[1:1000,]

#El paquete Udpipe proporciona modelos lingüísticos preentrenados para los respectivos idiomas
model <- udpipe_download_model(language = "english")
ud_eng <- udpipe_load_model(model$file_model)

# La siguiente instrucción tarda un poco
s <- udpipe_annotate(ud_eng,test$Review_Text)
x <- data.frame(s)

# RAKE

stats <- keywords_rake(x = x, term = "lemma", group = "doc_id", 
                       relevant = x$upos %in% c("NOUN", "ADJ"))
stats$key <- factor(stats$keyword, levels = rev(stats$keyword))
barchart(key ~ rake, data = head(subset(stats, freq > 3), 20), col = "red", 
         main = "RAKE - Palabras clave (Nombres y Adjetivos)", 
         xlab = "Rake")



### PROBLEMA 3

## Nueva Columna (Polarity --> 'neg' o 'pos')

dr <- dr %>%
  add_column(Polarity = if_else(.$Rating >= 4, 'pos', 'neg'))

# Extraemos solo las review y la nueva columna

dr_model <- dr[, c("Review_Text", "Polarity")]

# Semilla para elegir de forma random las 35000 opiniones dentro de las totales
set.seed(300)
id_train <- sample(1:42632, 35000, replace = FALSE)

mycorp <- corpus(dr_model$Review_Text)

mycorp$id_numeric <- 1:ndoc(mycorp)
mycorp$sentiment <- dr_model$Polarity

# Data cleaning
mytokens <- tokens(mycorp, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE)
mytokens <- tokens_wordstem(mytokens)
mytokens <- tokens_tolower(mytokens)
mytokens <- tokens_remove(mytokens, stopwords("english"))

mydfm <- dfm(mytokens)

# training set (elementos que contengan el id_train)
dfmat_training <- dfm_subset(mydfm, id_numeric %in% id_train)

# test set (elementos que NO contengan el id_train)
dfmat_test <- dfm_subset(mydfm, !id_numeric %in% id_train)

## Naive Bayes
tmod_nb <- textmodel_nb(dfmat_training, dfmat_training$sentiment)
summary(tmod_nb)


dfmat_matched <- dfm_match(dfmat_test, features = featnames(dfmat_training))

actual_class <- dfmat_matched$sentiment
predicted_class <- predict(tmod_nb, newdata = dfmat_matched)
tab_class <- table(actual_class, predicted_class)

## Confusion Matrix
confusionMatrix(tab_class, mode = "everything")

