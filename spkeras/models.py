import tensorflow as tf
from tensorflow.keras import backend as K
from .layers import  Regularizer, SpikeActivation, SpikingLayer, CFS,  Clip
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras import activations, Model

class SNN_Model(Model):
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        timestep = x.shape[1]
        loss = 0
        _loss = 0
        loss_mse_fn = tf.keras.losses.MeanSquaredError()
        loss_ce_fn = tf.keras.losses.CategoricalCrossentropy()

        trainable_vars = self.trainable_variables
        accum_gradient  = [tf.zeros_like(this_var) for this_var in trainable_vars]
        #accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        y_tot = 0
        loss_tot = 0
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            #y_pred = tf.reduce_mean(tf.nn.softmax(y_pred),axis=[1]) 
            #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            T = y_pred.shape[1]
            shape = [y_pred.shape[0]*y_pred.shape[1]]
            shape.extend(y_pred.shape[2:])
            y_pred = tf.reshape(y_pred,shape)
            y_pred = tf.math.softmax(y_pred)
            shape = [int(y_pred.shape[0]/T),T]
            shape.extend(y_pred.shape[1:])
            y_pred = tf.reshape(y_pred,shape)            
            y_pred = tf.reduce_mean(y_pred,axis=[1]) 
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #loss = self.compiled_loss(y, tf.nn.softmax(y_pred), regularization_losses=self.losses)
            
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        timestep = x.shape[1]
        # Compute predictions
        loss = 0
        loss_ce_fn = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            y_pred = self(x, training=False)
            #y_pred = tf.reduce_mean(tf.nn.softmax(y_pred),axis=[1]) 
            #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            T = y_pred.shape[1]
            shape = [y_pred.shape[0]*y_pred.shape[1]]
            shape.extend(y_pred.shape[2:])
            y_pred = tf.reshape(y_pred,shape)
            y_pred = tf.math.softmax(y_pred)
            shape = [int(y_pred.shape[0]/T),T]
            shape.extend(y_pred.shape[1:])
            y_pred = tf.reshape(y_pred,shape)            
            y_pred = tf.reduce_mean(y_pred,axis=[1]) 
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #loss = self.compiled_loss(y, tf.nn.softmax(y_pred), regularization_losses=self.losses)
            
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}    

class cnn_to_snn(object):
    def __init__(self,timesteps=256,thresholding=0.5,signed_bit=0,scaling_factor=1,method=1,amp_factor=100,percentile=100
                 ,epsilon = 0.001, spike_generator = False, spike_ext=0,noneloss=False,user_define=0):
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.method = method
        self.epsilon = epsilon
        self.amp_factor = amp_factor
        self.bit = signed_bit
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.use_bias = None
        self.spike_generator = spike_generator
        self.user_define = user_define
        self.percentile = percentile
        
    def __call__(self,mdl,x_train):
        for layer in mdl.layers:
            layer_type = type(layer).__name__
            if hasattr(layer, 'activation') and layer_type != 'Activation':
                use_bias = layer.use_bias
                break
        self.use_bias = use_bias        
        self.get_config()
        self.model = self.convert(mdl,x_train,                    
                                  thresholding = self.thresholding,
                                  scaling_factor = self.scaling_factor,
                                  method = self.method,
                                  percentile=self.percentile,
                                  timesteps=self.timesteps)
        
        return self
    
    def convert(self, mdl,x_train,thresholding=0.5,scaling_factor=1,method=0,timesteps=256,percentile=100):
        print('Start Converting...')
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        import numpy as np
        import json
        from tensorflow.keras.layers import Lambda

        old_mdl = mdl.to_json()
        new_mdl = mdl.to_json()

        old_mdl = json.loads(old_mdl)
        new_mdl = json.loads(new_mdl)

        model = model_from_json(mdl.to_json(),custom_objects = {"Regularizer": Regularizer,
                                                                "SpikingLayer": SpikingLayer,
                                                                "CFS": CFS,
                                                                "TA": TA,
                                                                "SNN_Model":SNN_Model,
                                                                "SpikingLayer":SpikingLayer,
                                                                "FindMax": FindMax,
                                                                "Clip":Clip,
                                                                "Constraints": Constraints,
                                                                "OutputRegularizer": OutputRegularizer,
                                                                "Lambda":  Lambda(lambda x: activations.relu(x,max_value=6)),
                                                                "PSNR": PSNR})
        
        model.set_weights(mdl.get_weights())

        epsilon = self.epsilon # defaut value in Keras
        amp_factor = self.amp_factor
        bit = self.bit - 1 if self.bit > 0 else 0
        method = 1 if bit == True else method 
        spike_ext = self.spike_ext
        user_define = self.user_define
        
        l,lmax = self.findlambda(model, x_train, batch_size=100,user_define=user_define,percentile=percentile)       
        layers=[]                            
        weights = []
        for layer in model.layers:
            _weights = layer.get_weights()
            layer_type = type(layer).__name__
            if _weights != [] and hasattr(layer, 'activation') :
                if not layer.use_bias:
                    _weights = [_weights[0],0] 
                _weights = [_weights,1] 
                weights.append(_weights)
            if layer_type == 'AveragePooling2D':
                weights.append([0,0])
            if layer_type == 'AveragePooling1D':
                weights.append([0,0])
            if layer_type == 'BatchNormalization':
                gamma,beta,mean,variance = layer.get_weights()
                weights[-1][0][0] = gamma/np.sqrt(variance+epsilon)*weights[-1][0][0] 
                weights[-1][0][1] = (gamma/np.sqrt(variance+epsilon)
                                                 *(weights[-1][0][1]-mean)+beta)
            #if layer_type == 'SpikingLayer':
            #    alpha = layer.get_weights()[0]
            #    weights[-1][0][1] = weights[-1][0][1]/alpha
            #    weights[-1][0][0] = weights[-1][0][0]/alpha

        vthr = []
        bias=[]
        kappa = amp_factor
        new_weights = []
        num=0
        last_num = len(weights) -1 
        for _weights in weights:
            if _weights[1] == 0:
                vthr.append(l[num].tolist()) 
                bias.append(0) 
                num += 1
                continue
                
            _weights = _weights[0]
            norm = 1
            if bit > 0 and num != 0 and num !=last_num:
            #if bit > 0:    
                norm = np.max(np.abs(_weights[0]))
                _weights[0] = _weights[0]*2**bit/norm
                _weights[0] = _weights[0].astype(int)   
                _weights[0] = _weights[0]/2**bit
            _bias = kappa*_weights[1]/lmax[num+1]
            _bias = _bias/norm
            bias.append(_bias.tolist())    
            _weights[0] = kappa*_weights[0]/l[num]
            _weights[1] = _weights[1]*0
            if isinstance(_weights[1], int):
                _weights.pop()
            new_weights.append(_weights)
            vthr.append(kappa/norm) 
            num += 1
        weights = new_weights
            
        print('Number of Spiking Layer:',len(vthr))
        print('threshold:',vthr)
        print('Note: threshold will be different when weight quantisation applied!')
        #n=2                           
        num = 0
        loc = 0
        functional = old_mdl['class_name'] == 'Functional' or old_mdl['class_name'] == 'SNN_Model'
        inbound_nodes = False
        for layer in old_mdl['config']['layers']:
            if layer['class_name'] == 'InputLayer':
                shape = layer['config']['batch_input_shape'] 
                if len(shape)>4:
                    _shape=[None]
                    _shape.extend(shape[2:])
                    shape = _shape
                
                layer['config']['batch_input_shape'] = shape
                layers.append(layer)
            if 'activation' in layer['config'] and layer['class_name'] != 'Activation':
                if functional:
                    inbound_nodes=layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]]
                    inbound_nodes = layer['config']['name']
                layers.append(layer)
                layers.append(self.spike_activation(threshold=vthr[loc],bias=bias[loc],scaling_factor=kappa,
                                                    inbound_nodes=inbound_nodes,name='spike_activation_'+str(loc)))   
                loc += 1
            if layer['class_name']=='Add':
                l_num = []
                for _inputs in layer['inbound_nodes'][0]:
                    txt = _inputs[0]
                    _txt = txt.split('_')
                    txt = 0 if _txt[0] == _txt[-1] else _txt[-1]
                    l_num.append(txt)

                l_gap = abs(int(l_num[0])-int(l_num[1]))*2
                identity=layers[-1-l_gap]['config']['name']
                residual=layers[-1]['config']['name']
                inbound_node=[[[identity, 0, 0, {}], [residual, 0, 0, {}]]]
                layer['inbound_nodes']=inbound_node
                layers.append(layer)
            if layer['class_name'] =='Flatten':
                if functional:
                    inbound_nodes=layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]]
                layers.append(layer)
            if layer['class_name'] =='AveragePooling2D':
                if functional:
                    inbound_nodes = layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                    inbound_nodes = layer['config']['name']
                layers.append(layer)
                layers.append(self.spike_activation(threshold=vthr[loc],bias=bias[loc],scaling_factor=kappa,
                                                    inbound_nodes=inbound_nodes,name='spike_activation_'+str(loc)))
                loc += 1
            if layer['class_name'] =='AveragePooling1D':
                if functional:
                    inbound_nodes = layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                    inbound_nodes = layer['config']['name']
                layers.append(layer)
                layers.append(self.spike_activation(threshold=vthr[loc],bias=bias[loc],scaling_factor=kappa,
                                                    inbound_nodes=inbound_nodes,name='spike_activation_'+str(loc)))
                loc += 1
            if layer['class_name'] =='MaxPooling2D':
                if functional:
                    inbound_nodes = layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                layers.append(layer)
            #if layer['class_name'] =='SpikingLayer':
            #    if functional:
            #        inbound_nodes = layers[-1]['config']['name']
            #        layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
            #    layers.append(layer)
            if layer['class_name'] =='Activation':
                if layer['config']['activation'] == 'softmax':
                    if functional:
                        inbound_nodes = layers[-1]['config']['name']
                        layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                        #inbound_nodes = layer['config']['name']    
                    layers.append(layer)
            num+=1
        new_mdl['config']['layers'] = layers
        inbound_nodes=layers[-1]['config']['name']
        new_mdl['config']['output_layers'] =  [[inbound_nodes, 0, 0]]
        new_mdl = json.dumps(new_mdl)
        new_model = model_from_json(new_mdl,
                                     custom_objects={'SpikeActivation':SpikeActivation,
                                                     'SpikingLayer':SpikingLayer,
                                                     'SNN_Model':SNN_Model,
                                                     'PSNR': PSNR})
        input_shape = model.layers[0].input_shape
        #new_model.build(input_shape)                            
        #new_model = keras.Model(inputs=inputs, outputs=outputs)

        m = 0
        for layer in new_model.layers:
            layer_type = type(layer).__name__ 
            if hasattr(layer, 'activation') and layer_type != 'Activation':
                layer.set_weights(weights[m])
                m += 1
        new_model.compile('adam', 'categorical_crossentropy', ['accuracy']) 
        print('New model generated!')
        return new_model
    
    def findlambda_v2(self,model,x_train):   
        import numpy as np
        #k = 0
        lmax = np.max(x_train) 
        l = []
        #if model.layers[0].name != 'input':
        l.append(lmax)
        print('Extracting Lambda...')#,end='')
        k = 0
        layer_num = len(model.layers)
        for layer in model.layers:
            layer_type = type(layer).__name__            
            if layer_type == 'Regularizer':
                lmax = layer.get_weights()
                l.append(lmax[0])
            if _name == 'softmax':
                layer = model.layers[-2]
                functor= K.function([model.layers[0].input], [layer.output])
                lmax = 0
                for n in range(x_train.shape[0]//batch_size):
                    a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                    #a =1 
                    _lmax = np.max(a)
                    lmax = max(lmax,_lmax)
                l.append(lmax)
                continue
            k += 1
        print('maximum activations:',l)
        return l    

    def spike_activation(self,threshold, bias,timesteps=255, spike_ext=0,scaling_factor=1,inbound_nodes=False,name='spike_activation'):
        layer_config = {'class_name': 'SpikeActivation',
         'config': {'name': 'spike_activation_1',
          'trainable': True,
          'dtype': 'float32',
          'timesteps': 255,
          'threshold': 1.0,
          'spike_ext': 0,
          'thresholding': 0.5,
          'noneloss': False,
          'scaling_factor': 1.0},
          'name': 'spike_activation_1'}    

        if not inbound_nodes:
            layer_config['name']=name
            layer_config['config']['name']=name
            layer_config['config']['timesteps']=timesteps
            layer_config['config']['threshold']=threshold
            layer_config['config']['bias']=bias
            layer_config['config']['spike_ext']=spike_ext
            layer_config['scaling_factor']=scaling_factor
        else:
            layer_config['name']=name
            layer_config['config']['name']=name
            layer_config['config']['timesteps']=timesteps
            layer_config['config']['threshold']=threshold
            layer_config['config']['bias']=bias
            layer_config['config']['spike_ext']=spike_ext
            layer_config['config']['scaling_factor']=scaling_factor 
            layer_config['inbound_nodes']=[[[inbound_nodes, 0, 0, {}]]] 
        return layer_config
    
    def findlambda(self,model,x_train,batch_size=100,percentile=100,user_define=False):  
        import numpy as np
        #k = 0
        if self.spike_generator:
            lmax = 1
        else:
            lmax = np.max(x_train) 
        l = []
        #if model.layers[0].name != 'input':
        l.append(lmax)
        print('Extracting Lambda...')#,end='')
        k = 0
        layer_num = len(model.layers)
        act_num_max = 0
        add_cnt = 2        
        for layer in model.layers:
            layer_type = type(layer).__name__
            if hasattr(layer, 'activation') and layer_type == 'Activation' or hasattr(layer, 'pool_size') and layer_type !='MaxPooling2D':
                IsAddLayer = type(model.layers[k-1]).__name__ == 'Add' 
                if IsAddLayer:
                    layer = model.layers[k-1]
                    print('{0}/{1}'.format(k,layer_num),end='')
                    print(layer.__class__.__name__) 
                    l_num = []
                    for _inputs in layer.input:
                        txt = _inputs.name
                        txt = txt.split('/')[0]
                        _txt = txt.split('_')
                        txt = 0 if _txt[0] == _txt[-1] else _txt[-1]
                        l_num.append(txt)
                    act_num_min = min(int(l_num[0]),int(l_num[1]))
                    l_gap = abs(int(l_num[0])-int(l_num[1]))
                    if not user_define:                       
                        functor= K.function([model.layers[0].input], [layer.output])
                        lmax = 0
                        __lmax = []
                        for n in range(x_train.shape[0]//batch_size):
                            a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                            #a =1 
                            _lmax = np.max(a) if percentile == 100 else np.percentile(a,percentile)
                            __lmax.append(_lmax)
                            lmax = np.max(__lmax) if percentile == 100 else np.mean(__lmax)
                    else:
                        lmax = user_define   
                        
                    AddIsNeibour = (act_num_min - act_num_max) == 1
                    if AddIsNeibour:
                        add_cnt += 1
                    else:
                        add_cnt = 2
                        
                    for cnt in range(add_cnt):
                        l[-1-cnt*l_gap] = lmax                        
                    k+=1
                    
                    act_num_max = max(int(l_num[0]),int(l_num[1]))
                    continue
                    
                print('{0}/{1}'.format(k,layer_num),end='')
                print(layer.__class__.__name__)

                if hasattr(layer, 'activation'):
                    try:
                        _name = layer.activation.__name__
                    except:
                        _name = None
                    if _name == 'softmax':
                        __n = k-1
                        if not hasattr(model.layers[__n], 'activation'):
                            __n -= 1
                        layer = model.layers[__n]
                        print(layer)
                        functor= K.function([model.layers[0].input], [layer.output])
                        lmax = 0
                        __lmax = []
                        for n in range(x_train.shape[0]//batch_size):
                            a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                            #a = 1 
                            #_lmax = np.max(a)
                            #lmax = max(lmax,_lmax)
                            _lmax = np.max(a)# if percentile == 100 else np.percentile(a,percentile)
                            __lmax.append(_lmax)
                            lmax = np.max(__lmax)# if percentile == 100 else np.mean(__lmax)                            
                        l.append(lmax)
                        k+=1
                        continue

                if not user_define:                       
                    functor= K.function([model.layers[0].input], [layer.output])
                    lmax = 0
                    __lmax = []
                    for n in range(x_train.shape[0]//batch_size):
                        a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                        #a = 1 
                        #_lmax = np.max(a)
                        #lmax = max(lmax,_lmax)
                        _lmax = np.max(a) if percentile == 100 else np.percentile(a,percentile)
                        __lmax.append(_lmax)
                        lmax = np.max(__lmax) if percentile == 100 else np.mean(__lmax)                              
                else:
                    lmax = user_define
                l.append(lmax)
                
            k += 1
        print('maximum activations:',l)
        new_l = []
        num = 0
        for k in range(len(l)):
            if k == 0:
                continue
            new_l.append(l[k]/l[k-1])
        print('normalisation factor:',new_l)        
        return [new_l,l]
    
    def SpikeCounter(self,x_train,timesteps=255,thresholding=0.5,scaling_factor=1,
                     spike_ext=0,batch_size=100,noneloss=False,mode=0,event_input=False):
        
        import numpy as np
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.chts_model(timesteps,thresholding,scaling_factor,spike_ext=spike_ext,noneloss=noneloss,verbose=False)
            
        self.get_config()    
        if event_input:
            x_train = x_train
        else:
            x_train = np.round(x_train*self.timesteps)        
        model = self.model
        
        cnt = []
        l = []
        print('Extracting Spikes...')#,end='')
        k = 0
        for layer in model.layers:
            #print('.',end='')
            layer_type = type(layer).__name__

            if layer_type == 'SpikeActivation':
                print(layer.__class__.__name__)
                functor= K.function([model.layers[0].input], [layer.output])
                _cnt = []
                lmax = 0
                if x_train.shape[0]<batch_size:
                    batch_size = x_train.shape[0]
                    
                for n in range(x_train.shape[0]//batch_size):
                    a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                    if mode:                   
                        __cnt = np.floor(a)                      
                    else:
                        __cnt = np.sum(a)
                    _lmax = np.max(a)
                    lmax = max(lmax,_lmax)   
                    _cnt.append(__cnt)
                    
                if mode:
                    scnt = []
                    _cnt = np.array(_cnt)
                    n = int(np.max(_cnt))+1
                    for m in range(n):
                        scnt.append(np.count_nonzero((_cnt == m)))
                    _cnt = scnt
                else:
                    _cnt=np.ceil(np.sum(_cnt)/x_train.shape[0])
                    
                l.append(lmax)
                cnt.append(_cnt)
            k += 1
        print('Max Spikes for each layer:',l)  
        print('Total Spikes for each layer:',cnt)
        return l,cnt     

    
    def NeuronNumbers(self,mode=0): 
        #mode: 0. count every layer; 1. not count average pooling
        import numpy as np
        model = self.model
        k = 0
        cnt = []
        s = []
        print('Extracting NeuronNumbers...')#,end='')
        for layer in model.layers:
            #print('.',end='')
            layer_type = type(layer).__name__
            if layer_type == 'Conv2D' or layer_type == 'Dense':
                print(layer.__class__.__name__)
                s.append(layer.weights[0].shape)
                            
            if layer_type == 'SpikeActivation': 
                print(layer.__class__.__name__)
                if hasattr(model.layers[k-1], 'pool_size') and mode == 1:
                    k +=1
                    continue
                    
                _cnt = np.prod(layer.output_shape[1:])
                cnt.append(_cnt)
            k += 1
        print('Total Neuron Number:',cnt)
        print('Done!')
        return s,cnt
           
    def evaluate(self,x_test,y_test,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False,sf=None,fix=0,spike_generator=False,
                event_input=False):
        import numpy as np
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.spike_generator = spike_generator
        self.chts_model(timesteps=timesteps,thresholding=thresholding,
                                     scaling_factor=scaling_factor,
                                     spike_ext=spike_ext,noneloss=noneloss,sf=sf,spike_generator=spike_generator)
        
        self.get_config()
        if not spike_generator:
            if event_input:
                _x_test = x_test
            else:
                _x_test = x_test*fix if fix > 0 else np.round(x_test*self.timesteps)
        else:
            _x_test = x_test

        if not hasattr(self.model.layers[0],'batch_size'):
            return self.model.evaluate(_x_test,y_test)
        else:
            BATCH_SIZE = 128 if self.model.layers[0].batch_size == None else self.model.layers[0].batch_size
            test_dataset = tf.data.Dataset.from_tensor_slices((_x_test,y_test))
            val_ds = test_dataset.batch(BATCH_SIZE,drop_remainder=True)            
            return self.model.evaluate(val_ds)

    def chts_model(self,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False,sf=None,spike_generator=False,verbose=True):
        #method: 0:threshold norm 1:weight norm 
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        mdl = self.model
        model = model_from_json(mdl.to_json(),
                                     custom_objects={'SpikeActivation':SpikeActivation,
                                                     'SpikingLayer':SpikingLayer,
                                                     'SNN_Model':SNN_Model,
                                                     'PSNR': PSNR})
        
        model.set_weights(mdl.get_weights())
        input_shape = model.layers[0].input_shape
        # Go through all layers, if it has a ReLU activation, replace it with PrELU
        if verbose:
            print('Changing model timesteps...')
        k = 0
        
        for layer in model.layers:
            layer_type = type(layer).__name__
            if layer_type == 'SpikeActivation':
                if k == 0:
                    layer.spike_generator = spike_generator
                layer.thresholding = thresholding
                layer.scaling_factor = scaling_factor # if sf == None else sf[k]
                layer.timesteps = timesteps   
                layer.spike_ext = spike_ext
                layer.noneloss = noneloss
                k += 1 
            
        new_model = model_from_json(model.to_json(),
                                     custom_objects={'SpikeActivation':SpikeActivation,
                                                     'SpikingLayer':SpikingLayer,
                                                     'SNN_Model':SNN_Model,
                                                     'PSNR': PSNR})
        #new_model.build(input_shape)
        m = 0
        for layer in new_model.layers:
            layer.set_weights(mdl.layers[m].get_weights())
            m += 1
        new_model.compile('adam', 'categorical_crossentropy', ['accuracy']) 
        del mdl
        if verbose:
            print('New model generated!')
        self.model = new_model
        #return new_model    
    
    def get_config(self):
        config = {'timesteps': int(self.timesteps),
                  'thresholding': self.thresholding,
                  'amp_factor':self.amp_factor, 
                  'signed_bit': self.bit,
                  'spike_ext':self.spike_ext,
                  'epsilon':self.epsilon,
                  'use_bias':self.use_bias,                               
                  'scaling_factor': self.scaling_factor,
                  'noneloss': self.noneloss,
                  'spike_generator': self.spike_generator,
                  'percentile': self.percentile,
                  'method':self.method
                  }
        return print(dict(list(config.items())))
       