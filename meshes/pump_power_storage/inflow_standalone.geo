Merge "inflow_3d.geo";
Coherence;

//offset = 0;
offset = 1230000;

Physical Surface("inlet", offset+82000) = {offset+72000};
Physical Surface("side_wall", offset+820013) = {offset+72001, offset+72002, offset+72005, offset+72006, offset+72010, offset+72011, offset+72012,offset+72015, offset+72009, offset+720018, offset+720019, offset+72004, offset+72008};
Physical Surface("bottom_wall", offset+820002) = {offset+72003, offset+72007, offset+720022, offset+72099};
Physical Surface("top_wall", offset+820003) = {offset+78004, offset+78007};
//Physical Surface("outlet", offset+820004) = {offset+720024};
Physical Volume("volume", offset+98007) = {offset+98001,offset+98002,offset+98003,offset+98004, offset+98006};
