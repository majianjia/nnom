from building import *

cwd = GetCurrentDir()

objs = []

objs += Glob('src/core/*.c')
objs += Glob('src/layers/*.c')
objs += Glob('src/backends/*.c')
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