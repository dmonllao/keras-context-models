import layer

# 1 hidden and fully connected nn.
def fc_1h(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Extra hidden layer.
def fc_2h(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.add_fc(base_layer, params, name='pre-hidden')
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Combinations of 'r' features.
def comb(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.Combinations(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

def comb_reg(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.Combinations(base_layer, params, reg=True)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features.
def inctx(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivity(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features.
def inctx_reg(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivity(base_layer, params, reg=True)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features + original inputs.
def inctx_extra(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivityAndOriginalActivity(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features + original inputs.
def inctx_extra_reg(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivityAndOriginalActivity(base_layer, params, reg=True)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Simple context / no-context separation.
def simple_separate_all(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_all_reg(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params, reg=True)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_1(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params, n_ctx_units=1)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_1_reg(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params, n_ctx_units=1, reg=True)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Separate inputs, single output.
def complex_separate(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitAllInputs(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output
