CC = g++
CDEFS   = 
CFLAGS	= -Wall -g -I/home/tgee/projects/c++/algo/src -I/usr/include/boost-1_40_0
LFLAGS	= -L/home/tgee/projects/c++/algo/lib -ldraw -lcluster -ldataset -liosmanip -lcolor -lgsl -llapack -lglut 
SRCS	= main.cpp 

TARGET	= fcm
OBJS	= $(SRCS:.cpp=.o)

default : all
all	: $(TARGET)

clean	:
	rm $(OBJS)

clobber	: 
	rm $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LFLAGS) -o $(TARGET)

.cpp.o	: 
	$(CC) -c $(CDEFS) $(CFLAGS) $< -o $*.o

