#include "MGenerator.h"

void getSilentMovesBlack(Position&pos,MoveListe&liste){
    uint32_t movers=pos.getMoversBlack();
    const uint32_t nocc =~(pos.BP|pos.WP);
    while(movers!=0){
        const uint32_t index =__tzcnt_u32(movers);
        movers &=movers-1;
        const uint32_t maske =1<<index;
        const uint32_t mask =maske &pos.K;
        uint32_t squares =(maske<<4)|((maske&MASK_L3)<<3)|((maske&MASK_L5)<<5);
        uint8_t pieceType =0;
        if(mask){
            squares|=(mask>>4)|((mask&MASK_R3)>>3)|((mask&MASK_R5)>>5);
            pieceType =1;
        }
        squares&=nocc;
        while(squares!=0){
            const uint32_t next=__tzcnt_u32(squares);
            squares&=squares-1;
            Move move(index,next);
            move.setPieceType(pieceType);
            liste.addMove(move);
        }
    }
}

void getSilentMovesWhite(Position& pos,MoveListe& liste){
    uint32_t movers=pos.getMoversWhite();
    const uint32_t nocc =~(pos.BP|pos.WP);
    while(movers!=0){
        const  uint32_t index =__tzcnt_u32(movers);
        movers &=movers-1;
        const uint32_t maske =1<<index;
        uint32_t squares =(maske>>4)|((maske&MASK_R3)>>3)|((maske&MASK_R5)>>5);
        const uint32_t mask =maske &pos.K;
        uint8_t pieceType=0;
        if(mask){
            squares|=(mask<<4)|((mask&MASK_L3)<<3)|((mask&MASK_L5)<<5);
            pieceType=1;
        }
        squares&=nocc;
        while(squares){
            const uint32_t next =__tzcnt_u32(squares);
            squares&=squares-1;
            Move move(index,next);
            move.setPieceType(pieceType);
            liste.addMove(move);
        }
    }
}


void getMoves(Board &board,MoveListe& sucessors){
   getMoves(*board.getPosition(),sucessors);
}

void getMoves(Position&pos, MoveListe& liste){
    if(pos.color==BLACK){
        addCapturesBlack(pos,liste);
        if(liste.moveCounter>0)
            return;

        getSilentMovesBlack(pos,liste);

    }else{
        addCapturesWhite(pos,liste);
        if(liste.moveCounter>0)
            return;
        getSilentMovesWhite(pos,liste);
    }
}

void getCaptures(Board&board, MoveListe& liste){
    if(board.getMover()==BLACK){
        addCapturesBlack(board.pStack[board.pCounter],liste);
    }else{
        addCapturesWhite(board.pStack[board.pCounter],liste);
    }
}


void addCapturesBlack(Position& pos, MoveListe&  liste) {
    uint32_t movers = pos.getJumpersBlack();
    while (movers) {
        const uint32_t index = __tzcnt_u32(movers);
        const uint32_t maske =1<<index;
        movers &=movers-1;
        if((maske&pos.K)!=0){
            pos.BP^=maske;
            addBlackTestKings(pos,index, maske,0, liste);
            pos.BP^=maske;
        }else{
            addBlackTestPawns(pos,index, maske,0, liste);
        }
    }
}

void addCapturesWhite(Position & pos, MoveListe&  liste) {
    uint32_t movers = pos.getJumpersWhite();
    while (movers) {
        const uint32_t index = __tzcnt_u32(movers);;
        const uint32_t maske = 1 << index;
        movers &=movers-1;
        if((maske&pos.K)!=0){
            pos.WP^=maske;
            addWhiteTestKings(pos,index, maske,0, liste);
            pos.WP^=maske;
        }else{
            addWhiteTestPawns(pos,index, maske,0, liste);
        }
    }
}

void addWhiteTestKings(Position& pos,uint32_t orig,uint32_t current,uint32_t captures,MoveListe& liste){
    uint32_t BP=pos.BP^(captures);
    const  uint32_t nocc = ~(BP|pos.WP);
    const uint32_t temp0 = (current >> 4) & BP;
    const uint32_t temp1 = (((current & MASK_R3) >> 3) | ((current & MASK_R5) >> 5)) & BP;
    const uint32_t temp2 = (((current) << 4)) & BP;
    const uint32_t temp3 = ((((current) & MASK_L3) << 3) | (((current) & MASK_L5) << 5)) & BP;
     uint32_t imed=temp0|temp1|temp2|temp3;

    uint32_t dest0=(((temp0 & MASK_R3) >> 3) | ((temp0 & MASK_R5) >> 5))&nocc;
    uint32_t dest1 =((temp1 >> 4))&nocc;
    uint32_t dest2=(((temp2 & MASK_L3) << 3) | ((temp2 & MASK_L5) << 5))&nocc;
    uint32_t dest3=(temp3 << 4)&nocc;
    uint32_t dest = dest0|dest1|dest2|dest3;

    imed&=(((dest0 & MASK_L3) << 3) | ((dest0 & MASK_L5) << 5) | (dest1 << 4)) | (((dest2 & MASK_R3) >> 3) | ((dest2 & MASK_R5) >> 5) | ((dest3 >> 4)));
    if(dest==0){
        addKingMove(orig,__tzcnt_u32(current),captures,liste);
    }
    while(dest){
        uint32_t destMask =dest&(-dest);
        uint32_t capMask =imed&(-imed);
        dest&=dest-1;
        imed&=imed-1;
        addWhiteTestKings(pos, orig,destMask,(captures|capMask), liste);
    }
}

void addWhiteTestPawns(Position& pos,uint32_t orig,uint32_t current,uint32_t captures,MoveListe& liste){
    const  uint32_t nocc = ~(pos.BP|pos.WP);
    const uint32_t temp0 = (current >> 4) & pos.BP;
    const uint32_t temp1 = (((current & MASK_R3) >> 3) | ((current & MASK_R5) >> 5)) & pos.BP;
    uint32_t imed =temp0|temp1;
    uint32_t dest=((((temp0 & MASK_R3) >> 3) | ((temp0 & MASK_R5) >> 5)) | ((temp1 >> 4)))&nocc;
    //removing the pieces that can not be captured from immediate
    imed&=((dest & MASK_L3) << 3) | ((dest & MASK_L5) << 5) | ((dest << 4));
    if(dest==0){
        addPawnMove(orig,__tzcnt_u32(current),captures,liste);
    }
    while(dest){
        uint32_t destMask =dest&(-dest);
        uint32_t capMask =imed&(-imed);
        dest&=dest-1;
        imed&=imed-1;
        addWhiteTestPawns(pos, orig,destMask,(captures|capMask), liste);
    }

}

///////////////////////////////////////////////////

void addBlackTestKings(Position& pos,uint32_t orig,uint32_t current,uint32_t captures,MoveListe& liste){
    uint32_t WP=pos.WP^(captures);
    const  uint32_t nocc = ~(WP|pos.BP);
    const uint32_t temp0 = (current >> 4) & WP;
    const uint32_t temp1 = (((current & MASK_R3) >> 3) | ((current & MASK_R5) >> 5)) & WP;
    const uint32_t temp2 = (((current) << 4)) & WP;
    const uint32_t temp3 = ((((current) & MASK_L3) << 3) | (((current) & MASK_L5) << 5)) & WP;
    uint32_t imed=temp0|temp1|temp2|temp3;
    uint32_t dest0=(((temp0 & MASK_R3) >> 3) | ((temp0 & MASK_R5) >> 5))&nocc;
    uint32_t dest1=((temp1 >> 4))&nocc;
    uint32_t dest2=(((temp2 & MASK_L3) << 3) | ((temp2 & MASK_L5) << 5))&nocc;
    uint32_t dest3=(temp3 << 4)&nocc;


    uint32_t dest = dest0|dest1|dest2|dest3;
    imed&=(((dest0 & MASK_L3) << 3) | ((dest0 & MASK_L5) << 5) | (dest1 << 4)) | (((dest2 & MASK_R3) >> 3) | ((dest2 & MASK_R5) >> 5) | ((dest3 >> 4)));
    if(dest==0){
        addKingMove(orig,__tzcnt_u32(current),captures,liste);
    }
    while(dest){
        uint32_t destMask =dest&(-dest);
        uint32_t capMask =imed&(-imed);
        dest&=dest-1;
        imed&=imed-1;
        addBlackTestKings(pos, orig,destMask,(captures|capMask), liste);
    }
}

void addBlackTestPawns(Position& pos,uint32_t orig,uint32_t current,uint32_t captures,MoveListe& liste){
    const  uint32_t nocc = ~(pos.BP|pos.WP);
    const uint32_t temp0 = (current << 4) & pos.WP;
    const uint32_t temp1 = (((current & MASK_L3) << 3) | ((current & MASK_L5) << 5)) & pos.WP;
    uint32_t imed =temp0|temp1;
    uint32_t dest=((((temp0 & MASK_L3) << 3) | ((temp0 & MASK_L5) << 5)) | ((temp1 << 4)))&nocc;
    imed&=((dest & MASK_R3) >>3) | ((dest & MASK_R5) >> 5) | ((dest >> 4));
    if(dest==0){
        addPawnMove(orig,__tzcnt_u32(current),captures,liste);
    }
    while(dest){
        uint32_t destMask =dest&(-dest);
        uint32_t capMask =imed&(-imed);
        dest&=dest-1;
        imed&=imed-1;
        addBlackTestPawns(pos, orig,destMask,(captures|capMask), liste);
    }
}



