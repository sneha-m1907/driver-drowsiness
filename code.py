   import os
   from keras.preprocessing import image
   import matplotlib.pyplot as plt 
   import numpy as np
   from keras.utils.np_utils import to_categorical
COMPUTER SCIENCE 
    import random,shutil
    from keras.models import Sequential
    from              keras.layers             import 
   Dropout,Conv2D,Flatten,Dense,       MaxPooling2D, 
   BatchNormalization
    from keras.models import load_model
    def                                 generator(dir, 
   gen=image.ImageDataGenerator(rescale=1./255), 
   shuﬄe=True,batch_size=1,target_size=(24,24),class
   _mode='categorical' ):
       return 
   gen.ﬂow_from_directory(dir,batch_size=batch_size,s
   huﬄe=shuﬄe,color_mode='grayscale',class_mode=
   class_mode,target_size=target_size)
    BS= 32
    TS=(24,24)
    train_batch=    generator('data/train',shuﬄe=True, 
   batch_size=BS,target_size=TS)
    valid_batch=    generator('data/valid',shuﬄe=True, 
   batch_size=BS,target_size=TS)
    SPE= len(train_batch.classes)//BS
    VS = len(valid_batch.classes)//BS
    print(SPE,VS)
    # img,labels= next(train_batch)

COMPUTER SCIENCE 
    # print(img.shape)
    model = Sequential([
       Conv2D(32, kernel_size=(3, 3), activation='relu', 
   input_shape=(24,24,1)),
       MaxPooling2D(pool_size=(1,1)),
       Conv2D(32,(3,3),activation='relu'),
       MaxPooling2D(pool_size=(1,1)),
    #32 convolution ﬁlters used each of size 3x3
    #again
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D(pool_size=(1,1)),
    #64 convolution ﬁlters used each of size 3x3
    #choose the best features via pooling
       
    #randomly turn neurons on and off to improve 
   convergence
       Dropout(0.25),
    #ﬂatten since too many dimensions, we only want 
   a classiﬁcation output
       Flatten(),
    #fully connected to get all relevant data
       Dense(128, activation='relu'),
    #one more dropout for convergence' sake :) 
       Dropout(0.5),

COMPUTER SCIENCE 
    #output a softmax to squash the matrix into output 
   probabilities
       Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_
   crossentropy',metrics=['accuracy'])
    model.ﬁt_generator(train_batch, 
   validation_data=valid_batch,epochs=15,steps_per_e
   poch=SPE ,validation_steps=VS)
    model.save('models/cnnCat2.h5', 
   overwrite=True)import cv2
                   MAIN CODE

    import os
    from keras.models import load_model
    import numpy as np
    from pygame import mixer
    import time
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    face = cv2.CascadeClassiﬁer('haar cascade ﬁles\
   haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassiﬁer('haar cascade ﬁles\
   haarcascade_lefteye_2splits.xml')

COMPUTER SCIENCE 
    reye = cv2.CascadeClassiﬁer('haar cascade ﬁles\
   haarcascade_righteye_2splits.xml')
    lbl=['Close','Open']
    model = load_model('models/cnncat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    while(True):
       ret, frame = cap.read()
       height,width = frame.shape[:2] 
       gray                 =     cv2.cvtColor(frame, 
   cv2.COLOR_BGR2GRAY)
       
       faces                                        = 
   face.detectMultiScale(gray,minNeighbors=5,scaleFa
   ctor=1.1,minSize=(25,25))
       left_eye = leye.detectMultiScale(gray)
       right_eye =  reye.detectMultiScale(gray)
       cv2.rectangle(frame, (0,height-50) , (200,height) , 

COMPUTER SCIENCE 
   (0,0,0) , thickness=cv2.FILLED )
       for (x,y,w,h) in faces:
           cv2.rectangle(frame, (x,y) , (x+w,y+h) , 
   (100,100,100) , 1 )
       for (x,y,w,h) in right_eye:
           r_eye=frame[y:y+h,x:x+w]
           count=count+1
           r_eye                                    = 
   cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
           r_eye = cv2.resize(r_eye,(24,24))
           r_eye= r_eye/255
           r_eye=  r_eye.reshape(24,24,-1)
           r_eye = np.expand_dims(r_eye,axis=0)
           rpred = model.predict_classes(r_eye)
           if(rpred[0]==1):
               lbl='Open' 
           if(rpred[0]==0):
               lbl='Closed'
           break
       for (x,y,w,h) in left_eye:
           l_eye=frame[y:y+h,x:x+w]
           count=count+1
           l_eye                                    = 
   cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  

COMPUTER SCIENCE 
           l_eye = cv2.resize(l_eye,(24,24))
           l_eye= l_eye/255
           l_eye=l_eye.reshape(24,24,-1)
           l_eye = np.expand_dims(l_eye,axis=0)
           lpred = model.predict_classes(l_eye)
           if(lpred[0]==1):
               lbl='Open'   
           if(lpred[0]==0):
               lbl='Closed'
           break
       if(rpred[0]==0 and lpred[0]==0):
           score=score+1
           cv2.putText(frame,"Closed",(10,height-20), 
   font, 1,(255,255,255),1,cv2.LINE_AA)
       # if(rpred[0]==1 or lpred[0]==1):
       else:
           score=score-1
           cv2.putText(frame,"Open",(10,height-20), font, 
   1,(255,255,255),1,cv2.LINE_AA)
       
           
       if(score<0):
           score=0   
       cv2.putText(frame,'Score:'+str(score),

COMPUTER SCIENCE 
   (100,height-20),              font,             1,
   (255,255,255),1,cv2.LINE_AA)
       if(score>15):
           #person is feeling sleepy so we beep the alarm
           cv2.imwrite(os.path.join(path,'image.jpg'),fram
   e)
           try:
               sound.play()
               
           except:  # isplaying = False
               pass
           if(thicc<16):
               thicc= thicc+2
           else:
               thicc=thicc-2
               if(thicc<2):
                   thicc=2
           cv2.rectangle(frame,(0,0),(width,height),
   (0,0,255),thicc) 
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()
    cv2.destroyAllWindows()
