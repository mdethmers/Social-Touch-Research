include <BOSL2/std.scad>

// Pins are widthxdepthxheight in size
pinWidth = 2;                      /* width of the pin as it passes through the mould */
pinDepth = 1;                     /* depth of the pin as it passes through the mould; height of the printed pin*/
pinHeight = 2;                    /* height of the pin outside of mould */

mouldThick = 5.5;                 /* Thickness of the plywood to stick through */
baseWidth = 3;                  /* Width of the base  */
baseDepth = 1;                  /* Depth  of the base; height of the printed base */
baseHeight = 1;                 /* Height of the base, stickout under the mould */
cutWidth = .3;                  /* Width of the cut that the wires fit in */
cutHeight = 1.5;                /* Cut depth below the top of the pin */

totalHeight = baseHeight + mouldThick + pinHeight; /* Total length of the pin */


module pin () {     
     union () {
          difference() {
               cube([pinWidth,totalHeight ,pinDepth],anchor=FRONT+BOTTOM); /* pin */
               back(totalHeight - cutHeight) cube([cutWidth,cutHeight,pinDepth],anchor=FRONT+BOTTOM); /* cutout */
//               back(totalHeight-cutHeight) cyl(d=.8,h=4,$fn=64);
               difference() {   /* Thinning */
                    back(baseHeight+mouldThick) cube([pinWidth,pinHeight,pinDepth], anchor=FRONT+BOTTOM);
                    cube([1.6, totalHeight, pinDepth],anchor=FRONT+BOTTOM);
               }
          }
          cube([baseWidth,baseHeight,baseDepth],anchor=FRONT+BOTTOM);

     }
}

dist = .2;

module pins() {
     pin();
     right(pinWidth+dist) back(totalHeight+baseHeight+dist) zrot(180) pin();
}

pins();

module production (R=10,C=10) {
     shift = 2*(pinWidth+dist);
     shift2 = totalHeight+baseHeight+2*dist;

     for (j = [0:R]) for (i = [0:C]) {
               back(j*shift2) right(i*shift) pins();
          }
}          
          
production(1, 1);
