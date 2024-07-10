package java_programs;
import java.util.*;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author derricklin
 */
public class DETECT_CYCLE {
     public static boolean detect_cycle(Node node) {
          Node hare = node;
          Node tortoise = node;

          while (hare != null && hare.getSuccessor() != null) {
               tortoise = tortoise.getSuccessor();
               hare = hare.getSuccessor().getSuccessor();

               if (hare == tortoise) {
                    return true;
               }
          }

          return false;
     }
}
