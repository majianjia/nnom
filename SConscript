from building import *

cwd = GetCurrentDir()

objs = []

objs += Glob('src/*.c')

try:
    # here for as
    Import('asenv')
    MODULES = asenv['MODULES']
    CPPPATH=['%s/inc'%(cwd), '%s/port/stdc'%(cwd)]
    asenv.Append(CPPPATH = CPPPATH)

except:
    # here for rt-thread
    CPPPATH=['%s/inc'%(cwd), '%s/port/rt-thread'%(cwd)]
    objs = DefineGroup('nnom', objs, 
            depend = ['PKG_USING_NNOM'],
            CPPPATH = CPPPATH)    

Return('objs')