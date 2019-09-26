
def result_info(y_pred, y_true, accuracy=None, loss=None):
    
    if accuracy:
        print('\nACCURACY: %.2f%%' % (accuracy * 100))

    if loss:
        print('LOSS: %f' % loss)
    
    print('\n======== SANITY CHECK ========')
    print(y_pred)
    print(y_true)
    print('==============================')