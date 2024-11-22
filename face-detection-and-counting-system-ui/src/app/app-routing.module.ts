import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomePageComponent } from './home-page/home-page.component';
import { ResultPageComponent } from './result-page/result-page.component';

const routes: Routes = [
  { 
    path: 'upload', 
    component: HomePageComponent 
  },
  { path: 'result', 
    component: ResultPageComponent
  },
  { path: '', 
    redirectTo: '/upload',
    pathMatch: 'full' 
  },
  {
    path: '**',
    redirectTo: '/upload',
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
