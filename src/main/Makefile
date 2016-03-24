CC = g++
LIBDIR  = ~/projects/c++/algo/lib
LIBTOOL = libtool
CDEFS   = -DNDEBUG
CFLAGS	= -Wall -march=athlon64 -O2 \
          -I/home/tgee/projects/c++/algo/src -I/usr/include/boost-1_40_0/ -I/usr/include/boost-sandbox 
LFLAGS	= 
SRCS	= km.cpp \
          fcm.cpp \
          scfcm.cpp \
          afcm.cpp \
          asfcm.cpp \
          gkfcm.cpp \
          pfcm.cpp \
          pgkfcm.cpp \
          partition.cpp \
          validation.cpp \
          hardening.cpp \
          clusters_gnuplot.cpp \
          clusters_visual.cpp 
#         fmdsc.cpp \
#          epgkfmdsc.cpp \

TARGET	= libcluster.la
.SUFFIXES : .lo
OBJS	= $(SRCS:.cpp=.lo)

default : all
all	: $(TARGET)

clean	:
	rm -f $(OBJS) $(TARGET)

clobber	: 
	rm $(TARGET)

depend  :
	makedepend -- $(CFLAGS) -- $(SRCS)

$(TARGET): $(OBJS)
	$(LIBTOOL) --mode=link $(CC) $(LFLAGS) -o $@ $(OBJS) -rpath $(LIBDIR)

.cpp.lo:
	$(LIBTOOL) --mode=compile $(CC) -c $(CDEFS) $(CFLAGS) $< -o $*.o 

install : $(TARGET)
	$(LIBTOOL) --mode=install cp $(TARGET) $(LIBDIR)
