import { animate, state, style, transition, trigger } from '@angular/animations';
import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-result-page',
  templateUrl: './result-page.component.html',
  styleUrl: './result-page.component.scss',
  animations: [
    trigger('floatAnimation', [
      state('float', style({
        transform: 'translateY(-10px)'
      })),
      state('rest', style({
        transform: 'translateY(0px)'
      })),
      transition('rest <=> float', animate('1.5s ease-in-out')),
    ])
  ]
})
export class ResultPageComponent {
  result: any;
  imageUrl: string | ArrayBuffer | null = '';
  animationState = 'rest';

  constructor(private router: Router) { }

  ngOnInit() {
    this.result = localStorage.getItem("result");
  }

  goBack() {
    this.router.navigate(['/upload']); // Adjust this route as needed
  }
}
