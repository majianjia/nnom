from building import *

cwd = GetCurrentDir()

objs = []

objs += Glob('src/*.c')
CPPPATH=['%s/inc'%(cwd), '%s/port'%(cwd)]

try:
    # here for as
    Import('asenv')
    MODULES = asenv['MODULES']
    
    asenv.Append(CPPPATH = CPPPATH)

except:
    # here for rt-thread
    objs = DefineGroup('nnom', objs, 
            depend = ['PKG_USING_NNOM'],
            CPPPATH = CPPPATH)    

Return('objs')