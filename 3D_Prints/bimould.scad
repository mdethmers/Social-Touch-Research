include <BOSL2/std.scad>

$fn=2*12;

scale = 1;


thickness = 3;
topR      = scale*200/2;
botR      = scale*340/2;
height    = scale*420;
mouldT    = 2;
sealR     = topR - scale*30;

botH = 2;
splitH = height/2 + 5;


module inner() {
     N=2*19;
     M=27;
     downangle = adj_opp_to_ang(height, botR-topR);
     diff () 
          cyl(r1=topR,r2=botR,h=height,anchor=BOT) {
          tag("remove") cyl(r1=topR-mouldT,r2=botR-mouldT,h=height) {
               position(TOP) down(.1) cyl(r=botR-mouldT,h=1,anchor=BOT);
               position(BOT) up(.1) cyl(r=sealR,h=20,anchor=TOP);
               for (j = [0:M-1])
                    for (i = [0:N-1])
                         position(BOT) up(15) up(j*(height-20)/(M-1)) zrot(360/N*(i+.5)) yrot(downangle) xrot(45) cube([400,2,1],anchor=LEFT);
          }
          tag("keep") position(BOT) difference() {
               cyl(r=topR, h=botH, anchor=BOT);
               down(.1) cyl(r=sealR,h=20,anchor=BOT);
          }
     }
}

module outer() {
     diff () 
          cyl(r1=topR+thickness+mouldT,r2=botR+thickness+mouldT,h=height,anchor=BOT) {
          position(BOT) cyl(r=topR+thickness+mouldT, h=botH, anchor=TOP);
          tag("remove") cyl(r1=topR+thickness,r2=botR+thickness,h=height) {
               position(TOP) down(.1) cyl(r=botR+thickness,h=1,anchor=BOT);
               position(BOT) up(.1) cyl(r=sealR-3,h=20,anchor=TOP);
          }
          tag("keep") position(BOT) difference() {
               cyl(r=sealR,h=botH,anchor=BOT);
               cyl(r=sealR-3,h=4*botH,anchor=CENTRE);
          }
     }
}

module splitInner() {
     union() {
          inner();
          intersection() {               
               difference() {
                    union() {
                         up(splitH) cyl(r=botR,h=2*botH);
                         N=3*2*19;
                         for (i=[0:N-1]) zrot(i*360/N) cube([botR,1,splitH], anchor=BOT+LEFT);
                    }
                    cyl(r1=botR,r2=topR-20,h=height,anchor=BOT);
               }
               cyl(r1=topR-1,r2=botR-1,h=height,anchor=BOT);
          }
     }
}

module splitOuter(R=botR-10) {
     union() {
          up(botH) outer();
          difference() {
               union() {
                    up(splitH) cyl(r=R,h=2*botH);
                    N=2*24;
                    for (i=[0:N-1]) zrot(i*360/N)
                    intersection () {
                         cube([R,2,splitH],anchor=BOT+LEFT);
                         cyl(r1=10,r2=R,h=splitH,anchor=BOT);
                    }
               }
               cyl(r1=topR+thickness,r2=botR+thickness,h=height,anchor=BOT);
          }
          
     }
}

module lower() {
     intersection() {
          cube([500,500,splitH],anchor=BOT);
          children();
     }
}

module upper() {
     down(splitH) intersection() {
          up(splitH) cube([500,500,500],anchor=BOT);
          children();
     }
}

module tabs(radius=130,N=6) {
     for (i=[0:N-1]) {
          zrot(i*360/N) right(radius) up(.1) cyl(r=3,h=2*botH+.2, anchor=CENTRE);
     }
}

module print() {
     upper() splitInner();
     left(2*botR+10) lower() splitInner();
     
     back(2*botR+10) upper() splitOuter();
     back(2*botR+10) left(2*botR+10) lower() splitOuter();
}
// print();
//left(2*botR+10) up(botH) outer();


module intop() {
     difference() {
          upper() splitInner();
          tabs();
     }
}

module inbot() {
     union() {
          lower() splitInner();
          up(splitH) tabs();
     }
}

module outtop () {
     difference() {
          upper() splitOuter();
          tabs(150);
     }
}

module outbot() {
     union() {
          lower() splitOuter();
          up(splitH) tabs(150);
     }
}


module assembled() {
     up(splitH) intop();
     inbot();
     up(splitH) outtop();
     outbot();
}

module exploded() {
     up(3*splitH+10) intop();
     up(2*splitH) inbot();
     up(splitH + 10) outtop();
     outbot();
}

//exploded(); 
assembled();

//back_half(500)

//inbot();
//intop();
//outbot();
//outtop();
