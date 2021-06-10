import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
#matplotlib.use('Agg')
import numpy as np

#plotting learning curve
def plot_history(histories, outputdir, key1='binary_crossentropy', key2='loss'):
     #def plot_history(histories, key='acc'):
     plt.figure(figsize=(6,5))

     for name, history in histories:
          fig1 = plt.figure()
          val = plt.plot(history.epoch, history.history['val_'+key1],
                    '--', label=name.title()+' Val')
          plt.plot(history.epoch, history.history[key1], color=val[0].get_color(),
               label=name.title()+' Train')
          plt.title(key1.replace('_',' ').title())
          plt.xlabel('Epochs')
          plt.ylabel(key1.replace('_',' ').title())
          plt.legend()
          #plt.show()
          fig1.savefig(outputdir + '/accuracy.pdf')
          plt.cla()

          fig2 = plt.figure()
          val = plt.plot(history.epoch, history.history['val_'+key2],
                    '--', label=name.title()+' Val')
          plt.plot(history.epoch, history.history[key2], color=val[0].get_color(),
               label=name.title()+' Train')
          plt.title(key2.replace('_',' ').title())
          plt.xlabel('Epochs')
          plt.ylabel(key2.replace('_',' ').title())
          plt.legend()
          #plt.show()
          fig2.savefig(outputdir + '/loss.pdf')
          plt.cla()

          np.save(outputdir + '/learning_curve.npy',[history.epoch, history.history[key1],history.history['val_'+key1],history.history[key2]],history.history['val_'+key2])

     plt.close(fig1)
     plt.close(fig2)
