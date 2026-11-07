"""Predicate-dependent input regions, separate from structural contexts."""
import hashlib,json


def references(expression):
    result=set()
    def visit(node):
        if not isinstance(node,dict):return
        if node['op']=='fact':result.add(node['args'][0])
        for arg in node['args']:visit(arg)
    visit(expression);return sorted(result)


def thresholds(expression):
    values=set()
    def visit(node):
        if isinstance(node,dict):
            for arg in node['args']:visit(arg)
        elif type(node) in (int,float):values.add(node)
    visit(expression);return sorted(values)


def region(expression,facts):
    constants=thresholds(expression);values={}
    for name in references(expression):
        if name not in facts:return None
        value=facts[name]
        if type(value) in (int,float) and constants:
            value=['eq',value] if value in constants else ['interval',sum(value>n for n in constants)]
        values[name]=value
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def actual_facts(entry_arguments,formal_names):
    result={};by_name={r['name']:r for r in entry_arguments or []}
    for index,name in enumerate(formal_names):
        item=by_name.get(name);prefix=f'input.{index}'
        if not item:continue
        result[prefix+'.is_tensor']=item['kind']=='tensor';result[prefix+'.is_complex']=item['is_complex']
        if item['kind']!='tensor':continue
        result[prefix+'.dtype']=item['dtype'];result[prefix+'.is_contiguous']=item.get('contiguous')
        shape=item.get('size');stride=item.get('stride')
        if shape is None or stride is None:continue
        result[prefix+'.rank']=len(shape)
        for attribute,values in (('size',shape),('stride',stride)):
            for axis,value in enumerate(values):
                if isinstance(value,str):
                    try:value=int(value)
                    except ValueError:continue
                result[f'{prefix}.{attribute}.{axis}']=value;result[f'{prefix}.{attribute}.{axis-len(values)}']=value
    return result
