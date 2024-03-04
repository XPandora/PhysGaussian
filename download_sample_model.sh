if [ ! -d "model" ]
then
    mkdir model
fi

gdown 1zic8PNxUsYESyrVZ8Ywy-FxpmQ01PGpe -O ./model/ficus_whitebg-trained.zip
unzip ./model/ficus_whitebg-trained.zip -d ./model